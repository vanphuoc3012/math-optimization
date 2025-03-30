def normal_approx(input_params):
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

  # Giới hạn bán vé (theo từng fare level)
  x_E1_max = int(alpha * S_E1)
  x_E2_max = int(alpha * S_E2)
  x_S1_max = int(alpha * S_S1)
  x_S2_max = int(alpha * S_S2)
  x_B1_max = int(alpha * S_B1)

  # ---- 2. Định nghĩa hàm xấp xỉ Normal cho Binomial(x, p) ----
  def normal_approx_binomial(x, p):
      """
      Xấp xỉ Binomial(x, p) bằng Normal(μ, σ^2), với:
        μ = x * p
        σ^2 = x * p * (1 - p)
      Sau đó làm tròn và clamp (0 <= show <= x).
      """
      mean = x * p
      var = x * p * (1 - p)
      if var <= 0:
          # Trường hợp x=0 hoặc p=0 hoặc p=1... => show = round(mean)
          return int(round(mean))

      std = np.sqrt(var)
      # Lấy mẫu normal
      sample = np.random.normal(loc=mean, scale=std)
      # Làm tròn và giới hạn
      show = int(round(sample))
      # clamp về [0, x]
      show = max(0, min(show, x))
      return show

  # ---- 3. Hàm mô phỏng lợi nhuận (Monte Carlo) với hàm xấp xỉ ----
  def simulate_profit_approx(x_E1, x_E2, x_S1, x_S2, x_B1, n_sims=2000):
      """
      Mô phỏng n_sims lần để ước lượng lợi nhuận trung bình.
      Số hành khách show-up xấp xỉ bằng Normal approximation cho Binomial.
      Tính bumped nếu show-up > sức chứa ghế, sau đó tính chi phí bồi thường.
      """
      # Doanh thu cố định (bán vé trước)
      revenue = (r_E1 * x_E1 +
                r_E2 * x_E2 +
                r_S1 * x_S1 +
                r_S2 * x_S2 +
                r_B1 * x_B1)

      total_profit = 0.0

      for _ in range(n_sims):
          # Lấy mẫu số hành khách đến (xấp xỉ Normal)
          show_E1 = normal_approx_binomial(x_E1, p_E1)
          show_E2 = normal_approx_binomial(x_E2, p_E2)
          show_S1 = normal_approx_binomial(x_S1, p_S1)
          show_S2 = normal_approx_binomial(x_S2, p_S2)
          show_B1 = normal_approx_binomial(x_B1, p_B1)

          # Tính số bị từ chối nếu vượt quá sức chứa ghế
          bumped_E1 = max(show_E1 - S_E1, 0)
          bumped_E2 = max(show_E2 - S_E2, 0)
          bumped_S1 = max(show_S1 - S_S1, 0)
          bumped_S2 = max(show_S2 - S_S2, 0)
          bumped_B1 = max(show_B1 - S_B1, 0)

          # Chi phí bồi thường
          comp_cost = (bumped_E1 * c_E1 +
                      bumped_E2 * c_E2 +
                      bumped_S1 * c_S1 +
                      bumped_S2 * c_S2 +
                      bumped_B1 * c_B1)

          profit_once = revenue - comp_cost
          total_profit += profit_once

      # Lợi nhuận trung bình sau n_sims lần mô phỏng
      return total_profit / n_sims

  # ---- 4. Vòng lặp tìm kiếm ngẫu nhiên (Monte Carlo Search) ----
  num_iterations = 3000

  best_solution = None
  best_profit = -float('inf')

  for _ in range(num_iterations):
      # Sinh ngẫu nhiên x_E1, x_E2, ... trong giới hạn
      xE1 = np.random.randint(0, x_E1_max + 1)
      xE2 = np.random.randint(0, x_E2_max + 1)
      xS1 = np.random.randint(0, x_S1_max + 1)
      xS2 = np.random.randint(0, x_S2_max + 1)
      xB1 = np.random.randint(0, x_B1_max + 1)

      # Kiểm tra ràng buộc overbooking
      if (xE1 + xE2) <= int(alpha * S_E) and \
        (xS1 + xS2) <= int(alpha * S_S) and \
        xB1 <= int(alpha * S_B):

          # Tính lợi nhuận kỳ vọng bằng mô phỏng (Normal Approx)
          avg_profit = simulate_profit_approx(xE1, xE2, xS1, xS2, xB1, n_sims=2000)

          # Cập nhật nghiệm tốt nhất
          if avg_profit > best_profit:
              best_profit = avg_profit
              best_solution = (xE1, xE2, xS1, xS2, xB1)

  # ---- 5. In kết quả ----
  print("=== Kết quả tìm kiếm Monte Carlo (Normal Approx) ===")
  print(f"Số vé bán (E1, E2, S1, S2, B1) = {best_solution}")
  print(f"Lợi nhuận mô phỏng trung bình ước tính = {best_profit:,.0f} VNĐ")
  return {
      "normal_approx_x_E1": best_solution[0],
      "normal_approx_x_E2": best_solution[1],
      "normal_approx_x_S1": best_solution[2],
      "normal_approx_x_S2": best_solution[3],
      "normal_approx_x_B1": best_solution[4],
      "normal_approx_profit": best_profit
  }
