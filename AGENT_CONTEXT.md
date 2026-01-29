# AGENT_CONTEXT.md - EAI_01 Fundamentos Matemáticos

> **Propósito**: Contexto estruturado para agentes de IA responderem questões sobre este módulo.
> **Última atualização**: Janeiro 2026

## RESUMO EXECUTIVO

**Módulo**: EAI_01 - Fundamentos Matemáticos para IA  
**Objetivo**: Estabelecer bases matemáticas (álgebra linear, vetores, regressão) para IA  
**Abordagem**: Implementação manual antes de bibliotecas  
**Nível**: Fundamentação (primeiro módulo do projeto)

---

## ESTRUTURA DE ARQUIVOS

```
EAI_01/
├── 01_Vetores_e_Transformações/
│   ├── vetores_basicos.ipynb          [Classes Vetor2D/3D, operações, visualizações]
│   ├── transformacoes_lineares.ipynb  [Matrizes de transformação, 4 tipos principais]
│   ├── cisalhamento.ipynb             [Animações FuncAnimation, grades transformadas]
│   └── transformacoes_deep.ipynb      [Rotações avançadas, composições]
├── 02_Regressão_Linear_Manual/
│   └── regressao_manual.ipynb         [Mínimos quadrados, MSE, visualização resíduos]
└── 03_Algebra_Linear/
    └── algebra_linear.ipynb           [Multiplicação matricial, autovalores, sistemas]
```

---

## NOTEBOOKS - CONTEXTO DETALHADO

### 1. vetores_basicos.ipynb

**Conceitos implementados**:
- Classe `Vetor2D`: x, y, magnitude(), normalizado(), produto_escalar()
- Classe `Vetor3D`: x, y, z, produto_vetorial()
- Operações: __add__, __sub__, __mul__ (escalar)

**Estruturas de dados**:
```python
Vetor2D(x, y)  # Exemplo: Vetor2D(3, 4)
Vetor3D(x, y, z)  # Exemplo: Vetor3D(1, 1, 1)
```

**Fórmulas principais**:
- Magnitude: √(x² + y²)
- Normalização: (x/mag, y/mag)
- Produto escalar: x₁·x₂ + y₁·y₂
- Produto vetorial (3D): (y₁z₂-z₁y₂, z₁x₂-x₁z₂, x₁y₂-y₁x₂)

**Visualizações**:
- `plt.quiver()` para vetores 2D
- `ax.quiver()` para vetores 3D
- Representação gráfica de soma vetorial

**Bibliotecas usadas**:
- `numpy` (cálculos)
- `matplotlib.pyplot` (gráficos 2D)
- `mpl_toolkits.mplot3d.Axes3D` (gráficos 3D)

**Casos de uso em IA**:
- Embeddings em NLP
- Representação de features
- Cálculo de distâncias (KNN)
- Similarity measures

---

### 2. transformacoes_lineares.ipynb

**Conceitos implementados**:
4 tipos de transformações lineares via matrizes

**Matrizes de transformação**:

| Transformação | Matriz 2x2 | Efeito |
|--------------|-----------|---------|
| Escala | `[[sx, 0], [0, sy]]` | Estica/encolhe eixos |
| Rotação | `[[cos θ, -sin θ], [sin θ, cos θ]]` | Gira em torno origem |
| Reflexão X | `[[1, 0], [0, -1]]` | Espelha eixo X |
| Reflexão Y | `[[-1, 0], [0, 1]]` | Espelha eixo Y |
| Cisalhamento H | `[[1, k], [0, 1]]` | Inclina horizontalmente |
| Cisalhamento V | `[[1, 0], [k, 1]]` | Inclina verticalmente |

**Operação fundamental**:
```python
v_transformed = matriz @ vetor  # Multiplicação matricial
```

**Propriedades preservadas**:
- Linhas retas permanecem retas
- Origem (0,0) fixa
- Proporcionalidade mantida

**Visualizações**:
- Vetores antes/depois da transformação
- Múltiplos vetores simultaneamente
- Uso de cores para diferenciar estados

**Bibliotecas usadas**:
- `numpy` (np.array, @, np.radians, np.cos, np.sin)
- `matplotlib.pyplot`

**Aplicações em IA**:
- Data augmentation (imagens)
- Transformações de features
- PCA (Principal Component Analysis)
- Autoencoders

---

### 3. cisalhamento.ipynb

**Conceitos implementados**:
- Cisalhamento horizontal/vertical
- Animações de transformações
- Visualização de grades transformadas

**Funções principais**:
```python
shear_matrix_horizontal(k)  # [[1, k], [0, 1]]
shear_matrix_vertical(k)    # [[1, 0], [k, 1]]
rotation_matrix(theta)      # Matriz de rotação
plot_grid_transformation(matrix, title)  # Plota grade
animate_transform(matrix_final, steps=30)  # Anima transformação
```

**Técnica de animação**:
- Interpolação linear entre matriz identidade e matriz final
- `T = (1-α)·I + α·M_final` onde α ∈ [0,1]
- `FuncAnimation` do matplotlib
- 30 frames padrão, 200ms intervalo

**Visualizações únicas**:
- Grade regular transformada (meshgrid)
- Animação HTML exportável (`ani.to_jshtml()`)
- Comparação lado a lado: grade original vs transformada

**Bibliotecas usadas**:
- `numpy` (linspace, meshgrid, vstack)
- `matplotlib.pyplot`, `matplotlib.animation.FuncAnimation`
- `IPython.display.HTML`

**Parâmetros de visualização**:
- Grade: 10x10 pontos, range [-3, 3]
- Cores: cinza (original), vermelho (transformado)
- Ajuste automático de limites dos eixos

---

### 4. transformacoes_deep.ipynb

**Conceitos implementados**:
- Aplicações avançadas de transformações
- Rotações em múltiplos ângulos
- Composição de transformações

**Exemplo de código**:
```python
angle = 30  # graus
R = rotation_matrix(angle)
v_rotated = R @ v
```

**Visualizações**:
- Comparação vetores originais vs rotacionados
- Múltiplos ângulos de rotação
- Figsize padrão: (10, 5)

**Bibliotecas usadas**:
- `numpy`
- `matplotlib.pyplot`

**Casos avançados**:
- Composição de transformações (rotação + escala)
- Transformações sequenciais
- Visualização de trajetórias

---

### 5. regressao_manual.ipynb

**Conceitos implementados**:
- Regressão linear por mínimos quadrados (manual)
- Cálculo de coeficientes sem bibliotecas
- Visualização de resíduos

**Equação do modelo**:
```
y = a·x + b
```

**Fórmulas dos coeficientes**:
```
a = [n·Σ(xy) - Σx·Σy] / [n·Σ(x²) - (Σx)²]
b = [Σy - a·Σx] / n
```

**Dataset exemplo**:
```python
x = [1.5, 1.6, 1.7, 1.8, 1.9]  # Altura (metros)
y = [45, 55, 65, 72, 85]       # Peso (kg)
# n = 5 pontos
```

**Métricas de avaliação**:
- MSE (Mean Squared Error) = Σ(y_real - y_pred)² / n
- Resíduos: distâncias verticais entre pontos e reta

**Implementação**:
```python
# Cálculo manual de somas
n = len(x)
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(x[i]*y[i] for i in range(n))
sum_x2 = sum(x[i]**2 for i in range(n))

# Coeficientes
a = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
b = (sum_y - a*sum_x) / n

# Previsão
y_pred = a*x + b
```

**Visualizações**:
- Scatter plot dos dados originais
- Reta de regressão sobreposta
- Linhas verticais mostrando resíduos
- Cores: azul (dados), vermelho (reta), tracejado (resíduos)

**Conceitos teóricos**:
- Mínimos quadrados: minimiza Σ(resíduos²)
- Interpretação de 'a': mudança em y por unidade de x
- Interpretação de 'b': valor de y quando x=0
- Relação linear: assume y ∝ x

**Bibliotecas usadas**:
- `numpy` (arrays, cálculos)
- `matplotlib.pyplot` (visualizações)

**Aplicações em ML**:
- Base para regressão linear em sklearn
- Fundamento de otimização (gradient descent)
- Avaliação de modelos (MSE, R²)

---

### 6. algebra_linear.ipynb

**Conceitos implementados**:
1. Operações com matrizes
2. Multiplicação matricial
3. Sistemas lineares
4. Autovalores e autovetores
5. Aplicações em IA

**Tópico 1: Conceitos Básicos**
- Vetores: arrays 1D
- Matrizes: arrays 2D
- Escalares: números reais

**Exemplo de vetores**:
```python
v1 = np.array([2, 1])
v2 = np.array([1, 2])
# Visualização com plt.quiver
```

**Tópico 2: Multiplicação de Matrizes**

**Regra dimensional**:
- A (m×n) @ B (n×p) = C (m×p)
- Número de colunas de A = número de linhas de B

**Exemplo prático**:
```python
A = [[1, 2, 3],
     [4, 5, 6]]  # 2x3

B = [[7, 8],
     [9, 10],
     [11, 12]]   # 3x2

C = A @ B  # 2x2
```

**Cálculo elemento (i,j)**:
- C[i,j] = Σ(A[i,k] · B[k,j]) para k=0 até n-1

**Tópico 3: Sistemas Lineares**
- Forma matricial: Ax = b
- Solução: x = A⁻¹b (quando A é inversível)
- Uso de `np.linalg.solve(A, b)`

**Tópico 4: Autovalores e Autovetores**

**Definição**:
- Av = λv (v é autovetor, λ é autovalor)
- Direções que não mudam sob transformação A

**Cálculo com NumPy**:
```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

**Interpretações**:
- Autovalores: fatores de escala
- Autovetores: direções principais
- Aplicação: PCA (redução de dimensionalidade)

**Tópico 5: Aplicações em IA**

| Conceito | Aplicação em IA |
|----------|-----------------|
| Multiplicação matricial | Camadas de redes neurais (W·x + b) |
| Transposição | Backpropagation |
| Produto escalar | Attention mechanisms |
| Autovalores | PCA, compressão de dados |
| Sistemas lineares | Mínimos quadrados |
| Normas | Regularização (L1, L2) |

**Operações NumPy essenciais**:
```python
A.T              # Transposição
A @ B            # Multiplicação
np.linalg.inv(A) # Inversa
np.linalg.det(A) # Determinante
np.linalg.eig(A) # Autovalores/vetores
np.linalg.norm(v) # Norma
```

**Visualizações**:
- Vetores como setas (quiver)
- Transformações matriciais
- Efeito de autovalores em direções

**Bibliotecas usadas**:
- `numpy` (operações matriciais)
- `matplotlib.pyplot` (visualizações)

**Conexões com Deep Learning**:
- Forward pass: multiplicações matriciais
- Batch processing: operações vetorizadas
- Gradientes: derivadas matriciais
- Inicialização de pesos: distribuições aleatórias

---

## DEPENDÊNCIAS TÉCNICAS

**Ambiente Python**:
- Python 3.12.7
- Conda environment: base

**Bibliotecas core**:
```python
import numpy as np                    # Operações numéricas
import matplotlib.pyplot as plt       # Gráficos 2D
from mpl_toolkits.mplot3d import Axes3D  # Gráficos 3D
from matplotlib.animation import FuncAnimation  # Animações
from IPython.display import HTML      # Renderização de animações
import math                           # Funções matemáticas básicas
```

**Versões específicas** (se necessário para reprodução):
- numpy: qualquer versão recente (>=1.20)
- matplotlib: >=3.3

---

## CONCEITOS MATEMÁTICOS - REFERÊNCIA RÁPIDA

### Vetores

**Magnitude (2D)**:
```
|v| = √(x² + y²)
```

**Produto escalar**:
```
u·v = u_x·v_x + u_y·v_y = |u||v|cos(θ)
```

**Normalização**:
```
v̂ = v/|v|
```

**Produto vetorial (3D)**:
```
u×v = (u_y·v_z - u_z·v_y, u_z·v_x - u_x·v_z, u_x·v_y - u_y·v_x)
```

### Matrizes de Transformação

**Rotação (θ em radianos)**:
```
R(θ) = [[cos θ, -sin θ],
        [sin θ,  cos θ]]
```

**Escala**:
```
S(sx, sy) = [[sx,  0],
             [ 0, sy]]
```

**Cisalhamento horizontal**:
```
Shx(k) = [[1, k],
          [0, 1]]
```

### Regressão Linear

**Coeficiente angular (a)**:
```
a = [n·Σ(xy) - Σx·Σy] / [n·Σ(x²) - (Σx)²]
```

**Intercepto (b)**:
```
b = (Σy - a·Σx) / n
```

**Erro Quadrático Médio**:
```
MSE = Σ(y_i - ŷ_i)² / n
```

---

## PERGUNTAS FREQUENTES - RESPOSTAS PARA AGENTES

**Q: Qual a diferença entre produto escalar e vetorial?**
A: Produto escalar retorna um número (u·v = escalar), indica similaridade/projeção. Produto vetorial retorna um vetor perpendicular aos dois originais (apenas 3D), indica área do paralelogramo.

**Q: Como calcular rotação de 45° no plano 2D?**
A: Usar matriz R(π/4) = [[cos(45°), -sin(45°)], [sin(45°), cos(45°)]] = [[√2/2, -√2/2], [√2/2, √2/2]]

**Q: Por que implementar regressão manualmente?**
A: Para entender o algoritmo dos mínimos quadrados antes de usar sklearn. Demonstra que é apenas álgebra: resolver sistema normal (X^T X)a = X^T y.

**Q: Qual a relação entre transformações lineares e deep learning?**
A: Cada camada de rede neural é uma transformação linear (Wx + b) seguida de não-linearidade. Compreender transformações matriciais é essencial para entender backpropagation.

**Q: Como autovalores ajudam em PCA?**
A: Autovalores da matriz de covariância indicam a variância explicada por cada componente principal. Autovetores são as direções desses componentes.

**Q: Diferença entre cisalhamento horizontal e vertical?**
A: Horizontal: move pontos na direção x proporcionalmente a y (matriz [[1,k],[0,1]]). Vertical: move pontos na direção y proporcionalmente a x (matriz [[1,0],[k,1]]).

**Q: Como normalizar um vetor?**
A: Dividir cada componente pela magnitude: v̂ = (x/|v|, y/|v|, z/|v|) onde |v| = √(x²+y²+z²)

**Q: O que significa "transformação linear preserva a origem"?**
A: T(0) = 0 sempre. Isso distingue transformações lineares de afins (que incluem translação).

---

## CÓDIGO DE REFERÊNCIA - SNIPPETS ÚTEIS

### Criar e visualizar vetor 2D
```python
v = Vetor2D(3, 4)
plt.quiver(0, 0, v.x, v.y, angles='xy', scale_units='xy', scale=1)
plt.xlim(-1, 5); plt.ylim(-1, 5)
plt.grid(True); plt.show()
```

### Aplicar transformação linear
```python
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
v_transformed = R @ v
```

### Regressão linear manual
```python
n = len(x)
a = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x**2) - sum(x)**2)
b = (sum(y) - a*sum(x)) / n
y_pred = a*x + b
mse = sum((y - y_pred)**2) / n
```

### Animar transformação
```python
def update(frame):
    alpha = frame / 30
    T = (1-alpha)*np.eye(2) + alpha*M_final
    # ... plotar vetores transformados por T
ani = FuncAnimation(fig, update, frames=31, interval=200)
```

---

## MÉTRICAS E RESULTADOS

**Dataset regressao_manual.ipynb**:
- Input: 5 pontos (altura, peso)
- Output: Reta y = ax + b
- MSE: calculado mas não especificado no notebook
- Visualização: reta + resíduos

**Transformações testadas**:
- Rotação: ângulos de 30°, 45°, 90°
- Escala: sx=2, sy=0.5 (exemplo típico)
- Cisalhamento: k=0.5, k=1 (parâmetros comuns)
- Reflexão: eixos X, Y, diagonal

---

## PRÓXIMOS MÓDULOS - CONEXÕES

**Para EAI_02 (Machine Learning)**:
- KNN usa distância euclidiana (magnitude de vetores diferença)
- Regressão linear estendida para múltiplas variáveis (vetores)
- Normalização de features (vetores normalizados)

**Para EAI_03 (Deep Learning)**:
- Camadas densas: multiplicação matricial (W @ x + b)
- Convolução: produto escalar em janelas deslizantes
- Backprop: derivadas de operações matriciais
- Inicialização: distribuições de matrizes de peso

**Para EAI_04 (NLP)**:
- Word embeddings: vetores em ℝ^d
- Similarity: cosine similarity (produto escalar normalizado)
- Transformações: projeções lineares em attention

**Para EAI_06 (Visão Computacional)**:
- Transformações geométricas em imagens (rotação, escala)
- Filtros de convolução: matrizes pequenas
- Operações pixel-wise: vetores RGB

---

## TAGS DE BUSCA

`#algebra-linear` `#vetores` `#matrizes` `#transformacoes-lineares` `#regressao-linear` `#numpy` `#matplotlib` `#visualizacao` `#matematica-ia` `#fundamentos` `#minimos-quadrados` `#autovalores` `#produto-escalar` `#rotacao` `#cisalhamento` `#animacao`

---

**Versão**: 1.0  
**Compatibilidade**: Agentes de IA com capacidade de processamento de estruturas markdown e código Python  
**Uso recomendado**: Contexto para responder perguntas específicas sobre implementações, conceitos, ou debugging do módulo EAI_01
