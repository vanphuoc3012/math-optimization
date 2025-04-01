def weighted_sum_method(input_params):
  import pulp

  def solve_airline_overbooking_MOO_with_constraints():
      """
      Ví dụ code: thêm ràng buộc x_{i,l} <= S_{i,l} và
      tổng vé bán <= alpha * tổng ghế,
      kèm mô hình multiobjective (doanh thu - beta*bồi thường).
      """
      # ---------------------------
      # 1. Định nghĩa tập (i,l)
      # ---------------------------
      # Ta đặt tên key là chuỗi, ví dụ: ('E','1'), ('E','2'), ...
      # Fare prices
      r_E1 = input_params['r_E1']  # Economy Standard
      r_E2 = input_params['r_E2']  # Economy Flexible
      r_S1 = input_params['r_S1']  # Special Economy Standard
      r_S2 = input_params['r_S2']  # Special Economy Flexible
      r_B1 = input_params['r_B1']  # Business Standard

      # Compensation costs (percentage of fare price)
      compensation_factor = input_params['boi_thuong'] * 0.01
      c_E1 = compensation_factor * r_E1
      c_E2 = compensation_factor * r_E2
      c_S1 = compensation_factor * r_S1
      c_S2 = compensation_factor * r_S2
      c_B1 = compensation_factor * r_B1

      # Seat capacity by class
      S_E = input_params['S_E']
      S_S = input_params['S_S']
      S_B = input_params['S_B']
      S_total = S_E + S_S + S_B

      # Seat allocation by fare level
      S_E1 = math.floor(input_params['S_E1'] * S_E * 0.01)
      S_E2 = S_E - S_E1
      S_S1 = math.floor(input_params['S_E2'] * S_S * 0.01)
      S_S2 = S_S - S_S1
      S_B1 = math.floor(S_B * input_params['S_B1'] * 0.01)

      # Show-up probabilities
      p_E1 = input_params['p_E1']
      p_E2 = input_params['p_E2']
      p_S1 = input_params['p_S1']
      p_S2 = input_params['p_S2']
      p_B1 = input_params['p_B1']

      # Overbooking factor
      alpha = input_params['alpha']
      T_max = math.floor(alpha * S_total)
      fare_types = [('E','1'), ('E','2'),
                    ('S','1'), ('S','2'),
                    ('B','1')]

      # ---------------------------
      # 2. Tham số
      # ---------------------------
      # Giá vé r_{i,l}
      r = {
          ('E','1'): r_E1,  # Economy Standard
          ('E','2'): r_E2,  # Economy Flexible
          ('S','1'): r_S1,  # Special Economy Standard
          ('S','2'): r_S2,  # Special Economy Flexible
          ('B','1'): r_B1   # Business Standard
      }

      # Xác suất xuất hiện p_{i,l}
      p = {
          ('E','1'): p_E1,
          ('E','2'): p_E2,
          ('S','1'): p_S1,
          ('S','2'): p_S2,
          ('B','1'): p_B1
      }

      # Sức chứa (hoặc giới hạn) cho từng (i,l)
      # Giả sử:
      #   Economy: 211 ghế, chia 60% Standard, 40% Flexible => (E,1)=126, (E,2)=84
      #   Special Economy: 35 ghế, chia 60%-40% => (S,1)=21, (S,2)=14
      #   Business: 28 ghế => (B,1)=28
      S_cap = {
          ('E','1'): S_E1,
          ('E','2'): S_E2,
          ('S','1'): S_S1,
          ('S','2'): S_S2,
          ('B','1'): S_B1
      }

      # Tổng ghế
      S_total = sum(S_cap.values())

      # Chi phí bồi thường = 200% giá vé
      c = {k: input_params['boi_thuong'] * 0.01*r[k] for k in fare_types}

      # ---------------------------
      # 3. Hàm con: Giải max [f(x) - beta*g(x)]
      # ---------------------------
      def solve_subproblem(beta):
          """
          Bài toán con:
            max  sum(r_{i,l} * x_{i,l}) - beta * sum(c_{i,l} * b_{i,l})
          s.t.
            x_{i,l} <= S_{i,l}                 (mới thêm)
            sum_{(i,l)} x_{i,l} <= alpha * sum_{(i,l)} S_{i,l}
            b_{i,l} >= p_{i,l}*x_{i,l} - S_{i,l}
            b_{i,l} >= 0
            x_{i,l} >= 0 và nguyên
          """
          prob = pulp.LpProblem("Overbooking_Subproblem", pulp.LpMaximize)

          # Tạo biến quyết định
          x = {k: pulp.LpVariable(f"x_{k[0]}{k[1]}",
                                  lowBound=0, cat=pulp.LpInteger)
              for k in fare_types}
          b = {k: pulp.LpVariable(f"b_{k[0]}{k[1]}",
                                  lowBound=0, cat=pulp.LpContinuous)
              for k in fare_types}

          # Hàm mục tiêu
          revenue_part = pulp.lpSum(r[k] * x[k] for k in fare_types)
          compcost_part = pulp.lpSum(c[k] * b[k] for k in fare_types)
          prob += revenue_part - beta*compcost_part

          # Ràng buộc: mỗi (i,l) không vượt capacity gốc
          for k in fare_types:
              prob += x[k] >= S_cap[k]

          # Ràng buộc: tổng vé bán <= alpha * tổng ghế
          prob += pulp.lpSum([x[k] for k in fare_types]) <= alpha * S_total

          # Ràng buộc b_{i,l} >= p_{i,l}*x_{i,l} - S_{i,l}
          for k in fare_types:
              prob += b[k] >= p[k]*x[k] - S_cap[k]
              prob += b[k] >= 0

          # Giải
          prob.solve(pulp.PULP_CBC_CMD(msg=0))

          # Lấy nghiệm
          x_sol = {k: x[k].varValue for k in fare_types}
          b_sol = {k: b[k].varValue for k in fare_types}
          return x_sol, b_sol

      # ---------------------------
      # 4. Tính f(x), g(x)
      # ---------------------------
      def compute_fg(x_sol):
          # f(x) = sum(r_{i,l} * x_{i,l})
          f_val = sum(r[k] * x_sol[k] for k in fare_types)
          # g(x) = sum(c_{i,l} * max(p_{i,l}*x_{i,l} - S_{i,l}, 0))
          g_val = 0
          for k in fare_types:
              bumped = max(p[k]*x_sol[k] - S_cap[k], 0)
              g_val += c[k]*bumped
          return f_val, g_val

      # ---------------------------
      # 5. Vòng lặp cập nhật beta
      # ---------------------------
      max_iter = 20
      beta = 0.0  # Khởi tạo

      for it in range(max_iter):
          x_sol, b_sol = solve_subproblem(beta)
          f_val, g_val = compute_fg(x_sol)

          # Tính sub_obj = f(x) - beta*g(x)
          sub_obj = f_val - beta*g_val

          # In kết quả vòng lặp
          print(f"--- Iteration {it} ---")
          print(f"beta = {beta:.6f}")
          print(f"f_val = {f_val:,.0f}")
          print(f"g_val = {g_val:,.0f}")
          print(f"sub_obj = {sub_obj:,.0f} (f - beta*g)")

          if g_val > 1e-9:
              new_beta = f_val / g_val
          else:
              # Nếu g_val=0 => dừng sớm (không phải bồi thường)
              print(f"Iteration {it+1}: g_val=0 => dừng.")
              break

          if abs(new_beta - beta) < 1e-6:
              beta = new_beta
              break
          beta = new_beta

      # Lấy nghiệm cuối
      f_final, g_final = compute_fg(x_sol)
      print("===== Kết quả cuối cùng Weighted Sum =====")
      print(f"  beta = {beta}")
      print("  Nghiệm x_{i,l}:")
      for k in fare_types:
          print(f"    x_{k} = {x_sol[k]:.2f}")
      print(f"  Tổng vé bán = {sum(x_sol.values()):.2f}")
      print(f"  Doanh thu f(x) = {f_final:,.0f} VND")
      print(f"  Bồi thường g(x) = {g_final:,.0f} VND")
      print(f"  Lợi nhuận (f-g) = {f_final - g_final:,.0f} VND")
      
      # ('E','1'), ('E','2'),('S','1'), ('S','2'), ('B','1')
      return {
        "x_E1": x_sol[('E','1')],
        "x_E2": x_sol[('E','2')],
        "x_S1": x_sol[('S','1')],
        "x_S2": x_sol[('S','2')],
        "x_B1": x_sol[('B','1')],
        "total_tickets_sold": int(sum(x_sol.values())),
        "profit": int(f_final - g_final),
        "comp_cost": int(g_final),
        "total_revenue":  int(f_final)
      }
  return solve_airline_overbooking_MOO_with_constraints()