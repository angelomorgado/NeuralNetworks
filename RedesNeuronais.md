# Redes neuronais - Ângelo Morgado
Resumos baseados [neste artigo](https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9)

## 1 - Neurónios

  - O **neurónio** é a unidade básica de uma rede neuronal. Basicamente, um neurónio recebe *inputs*, faz matemática (*does maths*) com tais inputs e produz um determinado *output*<br />
  ![neurónio](https://miro.medium.com/max/600/1*JRRC_UDsW1kDgPK3MW1GjQ.png)<br />
  
  Três coisas estão a acontecer aqui:<br />
  - Primeiro, cada input será **multiplicado por um peso *wn***, isto diz no fundo qual o input mais importante e o menos importante.
  - Segundo, esses inputs multiplicados por pesos irão ser **somados por uma bias *b***, que age da mesma forma que uma ordenada na origem.
  - Terceiro, esta soma irá passar por uma **função de ativação *y***.

Após estes três passos a função final deverá parecer-se com a seguinte:<br />
![função de ativação](https://miro.medium.com/max/700/1*9BFMXPkoAqN_EW7XTPvuGg.png)<br />

O propósito da função de ativação é transformar o input num output que seja previsivel e fácil de se utilizar. Uma função bastante usada para ativação é a **sigmoide**:<br />

  ![sigmoide](https://miro.medium.com/max/700/1*Ul8Yu_r8GKSFillzbPFrPQ.png)<br />

Esta função é boa porque apenas dá outputs de números entre 0 e 1, não importa qual o input dado, números muito próximos do -∞ ficam aproximados de 0 e números mais próximos de +∞ ficam aproximados de 1.<br />

### Exemplo de neurónios

Assumindo que temos 2 neurónios de input que usam a sigmoide como função de ativação e que tem os determinados parâmetros: **w = [0,1]** (é uma forma de escrever w1=0 e w2=1 na forma vetorial) e que **b=4**. <br/>
Agora, tendo o input **x = [2,3]** poderemos ter o nosso output:<br/>

*y = f((w1 * x1) + (w2 * x2) + b) = f((0 * 2) + (1 * 3) + 4)) = f(7) = 0.999*<br/>

O neurónio dá como output 0.999. O processo de passar os inputs para a frente para receber um output é chamado de **feedfoward**.
<br/>

Para cimentar o exemplo em programação foi feito [este programa](https://replit.com/@Morgado/Neuron-Example#main.py) em python utilizando a biblioteca numpy que simula o exercício anterior.<br 


***

## 2 - Combinar neurónios numa rede neuronal

  - Uma rede neuronal nada mais é que um conjunto de neurónios conectados uns aos outros por camadas. Eis um exemplo do que uma rede neuronal se poderá parecer: <br/>
  ![rede neuronal](https://victorzhou.com/77ed172fdef54ca1ffcfb0bba27ba334/network.svg)

  Esta rede neuronal tem **dois inputs**, **uma camada escondida** (*hidden layer*) com dois neurónios h1 e h2 e **uma camada de output** com 1 neurónio o1. Importante notar que os inputs do o1 são os outputs de h1 e h2, são esses que tornam isto uma rede.<br/>

  É possivel existirem *múltiplas camadas escondidas*, pois as camadas escondidas são aquelas que estão entre a camada de input e a camada de output.

### Exemplo de uma rede neuronal - feedforward

  - Usando a rede neuronal na imagem de cima e assumindo que todos os neurónios têm o mesmo peso **w = [0,1]**, bias **b=0** e a mesma função de ativação **sigmoide** (para manter a simplicidade do problema), façamos os cálculos para ver qual o output final *o1*, para isso temos de calcular os outputs da camada escondida *h1 e h2*:

*h1 = h2 = f(w * x + b) = f((w1 * x1) + (w2 * x2) + b) = f((0 ∗ 2) + (1 ∗ 3) + 0) = f(3) = 0.9526*

*o1 = f(w * [h1, h2] + b) = f(w1 * h1 + w2 * h2 + b) = f((0 * 0.9526) + (1 * 0.9526) + 0) = f(0.9526) = 0.7216*

O output final é 0.7216. Não importa quantas hidden layers existam, ou quantos neurónios existam em cada uma delas, o príncipio continua o mesmo, "alimentar" os inputs dos neuronios ao longo da rede para ter o output final.

O código mostrado [aqui](https://replit.com/@Morgado/Neural-network-example#main.py) exemplifica esta rede neuronal usando python e a sua biblioteca numpy.

<hr/>

## 3 - Treinar uma rede neuronal Parte 1

Tendo estes próximos dados como dados de treino:

| Name  | Weight(lb)  | Height(in)  |  Gender |
|---|---|---|---|
|Alice|133   | 65  |  F |
| Bob  |  160 | 72  | M |
| Charlie  | 152  |  70 |  M |
| Diana  | 120  |  60 |  F |

Iremos treinar a nossa rede mostrada no ponto anterior para prever o género de alguém dado o **peso** (Weight) e a **altura** (Height), representamos masculino com 0 e feminino com 1 e mudemos a tabela para ser mais fácil de usar pela rede.

| Name  | Weight(minus 135)  | Height(minus 66)  |  Gender |
|---|---|---|---|
|Alice|  -2  | -1  |  1 |
| Bob  |  25 | 6  | 0 |
| Charlie  | 17  |  4 |  0 |
| Diana  | -15  |  -6 |  1 |

*Neste caso os valores que foram subtraidos foram arbitrários, mas geralmente subtrai-se a média*

### Perda (Loss)

  - Antes de treinar a nossa rede precisamos ter uma forma de quantificar o quão "boa" é essa rede de forma a que possa ser "melhor". Para isso é que a **loss** existe.
  - Para este exemplo usaremos o **erro médio quadrático(MSE)**:<br/>
  *MSE = 1/n * &sum; ( y<sub>true</sub> - y<sub>pred</sub> ) <sup>2</sup>* <br/>
  Em que:
    - **n** é o número de amostras, que é 4 (Alice, Bob, Charlie, Diana);
    - **y** representa a variável a ser prevista, que é o género (Gender);
    - **y<sub>true</sub>** é o verdadeiro valor de y (a resposta "correta"). Por exemplo, para a Alice seria 1 (Female);
    - **y<sub>pred</sub>** é o valor previsto da variável, é o output da rede neuronal.
  
  - Quanto mais baixo o nosso valor da perda, melhor é a previsão. No fundo, treinar uma rede neuronal consiste em tentar diminuir a sua loss.
  - Para exemplificar o erro médio quadrático foi feito [este programa](https://replit.com/@Morgado/Mean-Squared-Error-MSE#main.py) em python.

---

## 4 - Treinar uma rede neuronal Parte 2

  - Sabemos que é possivel alterar os pesos e a bias de forma a influenciar a previsão, porém temos de alterar de forma a diminuir a loss. Para isso utilizamos cálculo.

  - Para simplificar o problema tomemos apenas em conta a Alice:

| Name  | Weight(minus 135)  | Height(minus 66)  |  Gender |
|---|---|---|---|
|Alice|  -2  | -1  |  1 |

  - Então, o erro médio quadrático é apenas o erro ao quadrado da Alice:

    - *MSE = 1/1 * &sum; ( y<sub>true</sub> - y<sub>pred</sub> ) <sup>2</sup>* <br/> = 
    *( y<sub>true</sub> - y<sub>pred</sub> ) <sup>2</sup>*<br/>
    = *( 1 - y<sub>pred</sub> ) <sup>2</sup>*

  - Outra forma de pensar acerca da perda é como uma função de pesos e biases. Para isso temos que dar um 'label' a cada peso e a cada bias na rede:<br/>
  ![rede2](https://victorzhou.com/27cf280166d7159c0465a58c68f99b39/network3.svg)

  - Após isso podemos escrever a loss como uma função de múltiplas variáveis:
  
    - *L(w1,w2,w3,w4,w5,w6,b1,b2,b3)*

  - A forma matemática de **encontrar o mínimo** de uma função é por aplicar derivadas, porém, se quisermos encontrar o mínimo de uma variável numa função multivariáveis deveremos utilizar **derivadas parciais**.

  - Para encontrar a derivada parcial de por exemplo o w1, deveremos utilizar uma regra matemática chamada de **chain rule**, que nos permite simplificar uma derivada ao multiplicarmos em cima e em baixo um determinado valor e em seguida separando em duas frações, tal e qual assim: *&part;L / &part;w<sub>1</sub> = &part;L * &part;y<sub>pred</sub> / &part;w<sub>1</sub> * &part;y<sub>pred</sub> = (&part;L / &part;y<sub>pred</sub>) * (&part;w<sub>1</sub> / &part;y<sub>pred</sub>)*.

  - Dever-se-á aplicar a chain rule até que seja possível fazer os cálculos da derivada e, assim, obter o mínimo da variável, no final a fórmula deverá parecer-se mais ou menos assim:
    *&part;L / &part;w<sub>1</sub> = (&part;L / &part;y<sub>pred</sub> ) * (&part;y<sub>pred</sub> / &part;h<sub>1</sub> ) * (&part;h<sub>1</sub> / &part;w<sub>1</sub> )*
  - O processo de calcular variáveis parciais ao trabalhar de frente para trás (usando a chain rule) é chamado de **backpropagation**.

  ### Treino: Stochastic Gradient Descent (Descida do gradiente)

  - Para sabermos como mudar o peso e bias para diminuir a loss usaremos um algorítmo chamado de **stochastic gradient descent (SGD)**, que basicamente é apenas esta equação de update:

    - *w<sub>1</sub> &larr; ( w<sub>1</sub> - η * &part;L / &part;w<sub>1</sub> )*
  
  - **η** é uma constante chamada de **learning rate** (ritmo de aprendizagem), que controla o quão rápido nós treinamos.

    - Se *&part;L / &part;w<sub>1</sub>* for **positivo**,  *w<sub>1</sub>* vai descer, o que faz com que a loss (*L*) desça.

    - Se *&part;L / &part;w<sub>1</sub>* for **negativo**,  *w<sub>1</sub>* vai subir, o que faz com que a loss (*L*) também desça.
  
  - Se fizermos este processo para cada peso e para cada bias na rede, a loss (L) vai começar a diminuir e a rede vai melhorar.

  - O **processo de treino** irá parecer-se com isto:
    1. Escolher **uma** amostra do dataset. Isto é o que faz a descida de gradiente estocástica, porém , em exemplos mais complexos não iremos utilizar só uma amostra, iremos utilizar um **batch** de n amostras de treino deixando de ser estocástica;
    2. Calcular todas as derivadas parciais da loss para cada peso e bias;
    3. Usar a equação de update para atualizar cada peso e bias;
    4. Voltar ao passo 1.

## 5 - Aplicar a rede neuronal em código

  - Agora que sabemos o necessário acerca de redes neuronais deveremos implementar em código tais redes neuronais

  - Aqui se encontram as informações do exercicio:

<br/>

![ex](https://victorzhou.com/27cf280166d7159c0465a58c68f99b39/network3.svg)

| Name  | Weight(minus 135)  | Height(minus 66)  |  Gender |
|---|---|---|---|
|Alice|  -2  | -1  |  1 |
| Bob  |  25 | 6  | 0 |
| Charlie  | 17  |  4 |  0 |
| Diana  | -15  |  -6 |  1 |


  - Poderá ver este exercício [aqui](https://replit.com/@Morgado/A-complete-neural-network#main.py) feito em python e utilizando o numpy
  
## 6 - Alguns termos extra




 