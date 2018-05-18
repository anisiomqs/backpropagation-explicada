import numpy as np

# 001
# 010
# 011
# 100
# 101
# 110
# 111

ingredientes = np.array([[0, 0, 0, 1, 1, 1, 1],
                         [0, 1, 1, 0, 0, 1, 1],
                         [1, 0, 1, 0, 1, 0, 1]])

habilidades_cozinheiros = np.array([[0.5, 0.5, 0.5],   # Cozinheiro 1
                                    [0.5, 0.5, 0.5],   # Cozinheiro 2
                                    [0.5, 0.5, 0.5],   # Cozinheiro 3
                                    [0.5, 0.5, 0.5],   # Cozinheiro 4
                                    [0.5, 0.5, 0.5],   # Cozinheiro 5
                                    [0.5, 0.5, 0.5],   # Cozinheiro 6
                                    [0.5, 0.5, 0.5]])  # Cozinheiro 7

pratos_corretos = np.identity(7)
grosseria = 1
erros = []

for sprint in range(1, 10):
    pratos_preparados = habilidades_cozinheiros.dot(ingredientes)  # 7x7
    custo = (1/14) * np.sum((pratos_corretos - pratos_preparados) ** 2)
    custo_matriz = (1/14) * ((pratos_corretos - pratos_preparados) ** 2)

    h_epsilon = habilidades_cozinheiros + 0.001
    p_epsilon = h_epsilon.dot(ingredientes)  # 7x7
    custo_epsilon = (1/14) * ((pratos_corretos - p_epsilon) ** 2)

    grad_approx = np.divide(p_epsilon - pratos_preparados, 0.001)

    erros.append(custo)
    # (3x7) Backpropagation - Retrospectiva
    culpa = - (1/7) * np.dot(pratos_corretos - pratos_preparados, ingredientes.T)

    novas_habilidades = habilidades_cozinheiros - grosseria * culpa  # Gradient Descent
    habilidades_cozinheiros = novas_habilidades

    if sprint % 10 == 0:
        print(custo)

teste = np.array([[0],
                  [1],
                  [1]])

r = habilidades_cozinheiros.dot(teste)
print(r)
print(np.argmax(r, axis=0) + 1)

