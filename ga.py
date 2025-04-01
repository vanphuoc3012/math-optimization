import math
import random
import numpy as np

# DEAP cho GA
from deap import base, creator, tools

def solve_use_ga(input, n_gen=50, pop_size=50):
  # ============================
  # 1. Tham số mô hình
  # ============================
  # Giá vé (VND)
  r_E1 = input['r_E1']  # Economy Standard
  r_E2 = input['r_E2']  # Economy Flexible
  r_S1 = input['r_S1']  # Special Economy Standard
  r_S2 = input['r_S2']  # Special Economy Flexible
  r_B1 = input['r_B1']  # Business Standard

  c_E1 = input['boi_thuong'] * r_E1 * 0.01
  c_E2 = input['boi_thuong'] * r_E2 * 0.01
  c_S1 = input['boi_thuong'] * r_S1 * 0.01
  c_S2 = input['boi_thuong'] * r_S2 * 0.01
  c_B1 = input['boi_thuong'] * r_B1 * 0.01

  # Số ghế mỗi class
  S_E = input['S_E']
  S_S = input['S_S']
  S_B = input['S_B']
  S_total = S_E + S_S + S_B

  # Phân bổ ghế cho từng fare level (60%-40% / 60%-40% / 100%)
  S_E1 = math.floor(input['S_E1'] * S_E * 0.01)   # Economy Standard
  S_E2 = S_E - S_E1              # Economy Flexible
  S_S1 = math.floor(input['S_E2'] * S_S * 0.01)   # Special Econ. Standard
  S_S2 = S_S - S_S1              # Special Econ. Flexible
  S_B1 = S_B  * input['S_B1'] * 0.01                   # Business Standard

  S_levels = [S_E1, S_E2, S_S1, S_S2, S_B1]

  # Xác suất show-up
  p_E1 = input['p_E1']
  p_E2 = input['p_E2']
  p_S1 = input['p_S1']
  p_S2 = input['p_S2']
  p_B1 = input['p_B1']
  p_levels = [p_E1, p_E2, p_S1, p_S2, p_B1]

  # Giá vé
  r_levels = [r_E1, r_E2, r_S1, r_S2, r_B1]

  # Bồi thường
  c_levels = [c_E1, c_E2, c_S1, c_S2, c_B1]

  # Overbooking factor
  alpha = input['alpha']
  T_max = math.floor(alpha * S_total)  # Tổng vé bán <= T_max

  N_FARE = 5  # 5 fare level: E1, E2, S1, S2, B1

  # ============================
  # 2. Hàm tính lợi nhuận
  # ============================
  def compute_profit(x_list):
      """
      x_list: [x_E1, x_E2, x_S1, x_S2, x_B1] (đảm bảo đã repair, sum <= T_max, x_l >= S_l)
      Return: profit (float)
      """
      # Tính revenue
      revenue = 0
      for i in range(N_FARE):
          revenue += r_levels[i] * x_list[i]

      # Tính chi phí bump: sum( c_l * max(p_l * x_l - S_l, 0) )
      comp_cost = 0
      for i in range(N_FARE):
          overshoot = p_levels[i] * x_list[i] - S_levels[i]
          if overshoot > 0:
              comp_cost += c_levels[i] * overshoot

      return revenue - comp_cost

  # ============================
  # 3. Hàm Repair cá thể
  # ============================
  def repair_ind(ind):
      """
      Sửa cá thể ind (list 5 gene) để thỏa mãn:
        1) x_l >= S_l
        2) sum(x_l) <= T_max
      """
      # Bước 1: Đảm bảo x_l >= S_l
      for i in range(N_FARE):
          if ind[i] < S_levels[i]:
              ind[i] = S_levels[i]

      # Bước 2: Đảm bảo tổng <= T_max (nếu vượt, ta scale xuống)
      total = sum(ind)
      if total > T_max:
          ratio = T_max / float(total)
          # scale x_l về S_l + ratio*(x_l - S_l)
          for i in range(N_FARE):
              base_val = S_levels[i]
              diff = ind[i] - base_val
              new_val = base_val + ratio * diff
              ind[i] = int(round(new_val))
          # có thể lặp lại check do int(round()) gây sai khác
          # ta lặp đến khi sum <= T_max hẳn
          # (Nhưng thường 1 lần đủ.)
          while sum(ind) > T_max:
              # Nếu vẫn > T_max, giảm dần
              exceed = sum(ind) - T_max
              # Chia đều exceed
              for i in range(N_FARE):
                  if ind[i] > S_levels[i]:
                      reduce_amt = min(exceed, ind[i] - S_levels[i])
                      ind[i] -= reduce_amt
                      exceed -= reduce_amt
                      if exceed <= 0:
                          break

      return ind

  # ============================
  # 4. Cấu trúc GA với DEAP
  # ============================
  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()

  # 4.1 Khởi tạo gene
  def random_gene(i):
      """Gene i random trong [S_levels[i], T_max]."""
      return random.randint(S_levels[i], T_max)

  def init_ind():
      """Khởi tạo 1 cá thể gồm 5 gene."""
      ind = [random_gene(i) for i in range(N_FARE)]
      # repair để đảm bảo feasible
      ind = repair_ind(ind)
      return ind

  toolbox.register("individual", tools.initIterate, creator.Individual, init_ind)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)

  # 4.2 Hàm đánh giá (evaluate)
  def eval_ind(ind):
      # Đảm bảo cá thể hợp lệ
      repair_ind(ind)
      # Tính profit
      profit = compute_profit(ind)
      return (profit,)

  toolbox.register("evaluate", eval_ind)

  # 4.3 Crossover & Mutation
  # - crossover: trao đổi gene với xác suất 0.5
  toolbox.register("mate", tools.cxUniform, indpb=0.5)

  # - mutation: random lại gene với prob=0.2
  def mut_uniform_custom(individual, indpb=0.2):
      for i in range(len(individual)):
          if random.random() < indpb:
              individual[i] = random.randint(S_levels[i], T_max)
      # repair
      repair_ind(individual)
      return (individual,)

  toolbox.register("mutate", mut_uniform_custom, indpb=0.2)

  # 4.4 Selection
  toolbox.register("select", tools.selTournament, tournsize=3)

  # ============================
  # 5. Vòng lặp tiến hóa
  # ============================
  def run_ga(n_gen=50, pop_size=50):
      # Khởi tạo quần thể
      pop = toolbox.population(n=pop_size)

      # Tính fitness ban đầu
      for ind in pop:
          ind.fitness.values = toolbox.evaluate(ind)

      for gen in range(n_gen):
          # Chọn cá thể (offspring)
          offspring = toolbox.select(pop, len(pop))
          offspring = list(map(toolbox.clone, offspring))

          # Lai ghép
          for i in range(0, len(offspring)-1, 2):
              if random.random() < 0.7:  # prob crossover
                  toolbox.mate(offspring[i], offspring[i+1])
                  del offspring[i].fitness.values
                  del offspring[i+1].fitness.values

          # Đột biến
          for mutant in offspring:
              if random.random() < 0.3:  # prob mutation
                  toolbox.mutate(mutant)
                  del mutant.fitness.values

          # Tính lại fitness cho cá thể mới
          invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
          for ind in invalid_ind:
              ind.fitness.values = toolbox.evaluate(ind)

          # Tạo quần thể thế hệ kế
          pop[:] = offspring

          # Thống kê
          fits = [ind.fitness.values[0] for ind in pop]
          best_fit = max(fits)
          avg_fit = sum(fits)/len(fits)
          print(f"Gen {gen+1}: Best={best_fit:,.0f} | Avg={avg_fit:,.0f}")

      # Kết thúc
      best_ind = tools.selBest(pop, 1)[0]
      revenue = best_ind[0] * r_levels[0] + best_ind[1] * r_levels[1] + best_ind[2] * r_levels[2] + best_ind[3] * r_levels[3] + best_ind[4] * r_levels[4]
      
      comp_cost = sum(
          max(p_levels[i] * best_ind[i] - S_levels[i], 0) * c_levels[i]
          for i in range(N_FARE)
      )
      return {
        "x_E1": best_ind[0],
        "x_E2": best_ind[1],
        "x_S1": best_ind[2],
        "x_S2": best_ind[3],
        "x_B1": best_ind[4],
        "total_tickets_sold": sum(best_ind),
        "profit": best_ind.fitness.values[0],
        "comp_cost": comp_cost,
        "total_revenue":  revenue
      }

  return run_ga(n_gen=n_gen, pop_size=pop_size)