def random_search(input_params):
  import numpy as np

   # ---- 1. Khai báo dữ liệu bài toán ----
  # Giá vé (VNĐ)
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

  # ===== 2. Định nghĩa hàm mục tiêu theo công thức trong hình =====
  def compute_profit(xE1, xE2, xS1, xS2, xB1):
      """
      Tính Profit = Revenue - CompensationCost
      = sum(r_i * x_i) - sum(max(p_i * x_i - S_i, 0) * c_i).
      """
      # (a) Tổng doanh thu từ việc bán vé
      revenue = (r_E1 * xE1 +
                r_E2 * xE2 +
                r_S1 * xS1 +
                r_S2 * xS2 +
                r_B1 * xB1)

      # (b) Chi phí bồi thường = sum( max( p_i * x_i - S_i, 0 ) * c_i )
      comp_cost = 0.0
      comp_cost += max(p_E1 * xE1 - S_E1, 0) * c_E1
      comp_cost += max(p_E2 * xE2 - S_E2, 0) * c_E2
      comp_cost += max(p_S1 * xS1 - S_S1, 0) * c_S1
      comp_cost += max(p_S2 * xS2 - S_S2, 0) * c_S2
      comp_cost += max(p_B1 * xB1 - S_B1, 0) * c_B1

      return revenue - comp_cost

  # ===== 3. Áp dụng Random Search để tìm nghiệm tốt =====

  num_iterations = 300000  # Số lần lặp tìm kiếm
  best_profit = -float('inf')
  best_solution = None

  # Ràng buộc:
  #   0 <= x_{E1} <= S_E1
  #   0 <= x_{E2} <= S_E2
  #   0 <= x_{S1} <= S_S1
  #   0 <= x_{S2} <= S_S2
  #   0 <= x_{B1} <= S_B1
  #   x_E1 + x_E2 + x_S1 + x_S2 + x_B1 <= alpha * S_total

  max_tickets = int(alpha * S_total)  # ví dụ: int(1.2 * 274) = 328

  for _ in range(num_iterations):
      # (a) Sinh ngẫu nhiên các x_{E1}, x_{E2}, ...
      xE1 = np.random.randint(0, S_E1 + 1)
      xE2 = np.random.randint(0, S_E2 + 1)
      xS1 = np.random.randint(0, S_S1 + 1)
      xS2 = np.random.randint(0, S_S2 + 1)
      xB1 = np.random.randint(0, S_B1 + 1)

      # (b) Kiểm tra ràng buộc tổng vé <= alpha * S_total
      total_tickets = xE1 + xE2 + xS1 + xS2 + xB1
      if total_tickets <= max_tickets:
          # (c) Tính hàm mục tiêu (Profit)
          profit = compute_profit(xE1, xE2, xS1, xS2, xB1)

          # (d) Cập nhật nghiệm tốt nhất
          if profit > best_profit:
              best_profit = profit
              best_solution = (xE1, xE2, xS1, xS2, xB1)

  # ===== 4. In kết quả =====
  print("=== Kết quả Random Search với hàm mục tiêu trong hình ===")
  print(f"Giải pháp tốt nhất (x_E1, x_E2, x_S1, x_S2, x_B1) = {best_solution}")
  print(f"Lợi nhuận tối đa ước tính = {best_profit:,.0f} VNĐ")
  return {
      "random_search_x_E1": best_solution[0],
      "random_search_x_E2": best_solution[1],
      "random_search_x_S1": best_solution[2],
      "random_search_x_S2": best_solution[3],
      "random_search_x_B1": best_solution[4],
      "random_search_profit": best_profit
  }