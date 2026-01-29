# EAI_01 - Fundamentos Matemáticos para IA

## 📚 Sobre este Módulo

Este módulo estabelece as bases matemáticas essenciais para entender e construir modelos de Inteligência Artificial. Aqui você aprenderá os conceitos fundamentais de álgebra linear, operações vetoriais e regressão linear através de uma abordagem **prática e visual**, implementando tudo do zero antes de usar bibliotecas prontas.

## 🎯 Objetivos de Aprendizagem

Ao finalizar este módulo, você será capaz de:

- ✅ Compreender e manipular vetores em 2D e 3D
- ✅ Aplicar transformações lineares usando matrizes
- ✅ Implementar regressão linear manualmente
- ✅ Visualizar geometricamente operações matemáticas
- ✅ Entender a matemática por trás dos modelos de IA

## 📂 Estrutura do Módulo

```
EAI_01_Fundamentos_Matemática_para_IA/
├── 01_Vetores_e_Transformações/
│   ├── vetores_basicos.ipynb          # Fundamentos de vetores 2D/3D
│   ├── transformacoes_lineares.ipynb  # Rotação, escala, reflexão
│   ├── cisalhamento.ipynb             # Cisalhamento e animações
│   └── transformacoes_deep.ipynb      # Aplicações avançadas
├── 02_Regressão_Linear_Manual/
│   └── regressao_manual.ipynb         # Implementação do zero
└── 03_Algebra_Linear/
    └── algebra_linear.ipynb           # Matrizes, sistemas lineares, autovalores
```

## 📖 Conteúdo Detalhado

### 01 - Vetores e Transformações

#### **vetores_basicos.ipynb**
Introdução aos conceitos fundamentais de vetores:
- Representação de vetores em 2D e 3D
- Cálculo de magnitude e normalização
- Operações vetoriais: soma, subtração, produto escalar
- Visualização gráfica de vetores
- **Conceito-chave**: Vetores como entidades com direção e magnitude

#### **transformacoes_lineares.ipynb**
Exploração de transformações geométricas usando matrizes:
- Rotação de vetores e objetos
- Escala e redimensionamento
- Reflexão em diferentes eixos
- Cisalhamento horizontal e vertical
- **Conceito-chave**: Matrizes como operadores de transformação

#### **cisalhamento.ipynb**
Foco em cisalhamento com visualizações avançadas:
- Implementação de funções de cisalhamento
- Animações de transformações
- Visualização de grade transformada
- **Conceito-chave**: Transformações progressivas e interpolação

#### **transformacoes_deep.ipynb**
Aplicações avançadas de transformações:
- Combinação de múltiplas transformações
- Rotações em diferentes ângulos
- Visualizações complexas
- **Conceito-chave**: Composição de transformações

### 02 - Regressão Linear Manual

#### **regressao_manual.ipynb**
Implementação completa de regressão linear sem bibliotecas:
- Cálculo manual dos coeficientes (inclinação e intercepto)
- Fórmulas dos mínimos quadrados
- Previsão de valores
- Cálculo e visualização de resíduos
- Avaliação com Erro Quadrático Médio (MSE)
- **Conceito-chave**: Ajuste de modelo aos dados

**Exemplo prático**: Previsão de peso baseado em altura
- Dataset: 5 pontos (altura vs peso)
- Equação: y = a·x + b
- Visualização: reta de regressão e resíduos

### 03 - Álgebra Linear

#### **algebra_linear.ipynb**
Fundamentos completos de álgebra linear para IA:
- Operações com matrizes (multiplicação, transposição)
- Resolução de sistemas lineares
- Autovalores e autovetores
- Transformações lineares com matrizes
- Aplicações em Machine Learning e Deep Learning
- **Conceito-chave**: Matrizes como representação de dados e transformações

**Tópicos cobertos**:
1. Conceitos básicos (vetores, matrizes, escalares)
2. Multiplicação de matrizes
3. Sistemas lineares
4. Autovalores e autovetores
5. Aplicações práticas em IA

## 🚀 Como Usar Este Módulo

### Pré-requisitos

```bash
# Bibliotecas necessárias
numpy
matplotlib
mpl_toolkits (para gráficos 3D)
```

### Ordem Recomendada de Estudo

1. **Comece com vetores_basicos.ipynb** - Base fundamental
2. **Prossiga para transformacoes_lineares.ipynb** - Aplicação prática
3. **Explore cisalhamento.ipynb** - Visualizações avançadas
4. **Aprofunde com transformacoes_deep.ipynb** - Casos complexos
5. **Aplique tudo em regressao_manual.ipynb** - Primeiro modelo de ML
6. **Consolide com algebra_linear.ipynb** - Visão geral e aplicações

### Executando os Notebooks

```bash
# Clone o repositório
git clone [seu-repositorio]

# Entre no diretório
cd EAI_01_Fundamentos_Matemática_para_IA

# Inicie o Jupyter
jupyter notebook
```

## 💡 Conceitos-Chave Aprendidos

### Matemática Fundamental
- **Vetores**: Direção + Magnitude
- **Matrizes**: Transformações lineares
- **Produto escalar**: Projeção e similaridade
- **Normalização**: Vetores unitários

### Transformações Lineares
- **Rotação**: Preserva distâncias e ângulos
- **Escala**: Estica ou comprime
- **Reflexão**: Espelha em eixos
- **Cisalhamento**: Inclina sem rotação

### Regressão Linear
- **Objetivo**: Minimizar erros quadráticos
- **Coeficientes**: Calculados por mínimos quadrados
- **Previsão**: Linha que melhor se ajusta aos dados
- **Avaliação**: MSE (Erro Quadrático Médio)

### Álgebra Linear em IA
- **Representação de dados**: Como matrizes/vetores
- **Redes neurais**: Multiplicação matricial
- **Transformações**: Processamento de imagens/texto
- **Sistemas lineares**: Resolução eficiente

## 🔗 Conexão com Próximos Módulos

Os conceitos aprendidos aqui são fundamentais para:

- **EAI_02 (Machine Learning)**: Algoritmos como KNN usam distâncias vetoriais
- **EAI_03 (Deep Learning)**: Redes neurais são multiplicações matriciais
- **EAI_04 (NLP)**: Word embeddings são vetores em espaço multidimensional
- **EAI_06 (Visão Computacional)**: Transformações em imagens

## 📝 Notas Importantes

- **Aprenda fazendo**: Execute cada célula e modifique os parâmetros
- **Visualize**: Os gráficos são essenciais para compreensão
- **Implemente do zero**: Evite bibliotecas prontas nesta fase
- **Experimente**: Teste com seus próprios dados

## 🎓 Recursos Complementares

### Para Aprofundamento
- **3Blue1Brown**: Série "Essence of Linear Algebra" (YouTube)
- **Khan Academy**: Álgebra Linear
- **MIT OCW**: Linear Algebra (Gilbert Strang)

### Livros Recomendados
- "Linear Algebra and Its Applications" - Gilbert Strang
- "Mathematics for Machine Learning" - Marc Peter Deisenroth

## ✅ Checklist de Progresso

- [ ] Compreendeu operações básicas com vetores
- [ ] Implementou transformações lineares
- [ ] Criou animações de transformações
- [ ] Calculou regressão linear manualmente
- [ ] Entendeu autovalores e autovetores
- [ ] Conectou conceitos com aplicações em IA

## 🤝 Contribuindo

Encontrou um erro ou tem uma sugestão? Abra uma issue ou envie um pull request!

---

**Próximo Módulo**: [EAI_02 - Machine Learning](../EAI_02_Machine_Learning)

**Anterior**: Início do Projeto

---

*Desenvolvido como parte do projeto "Especialista em IA"*
