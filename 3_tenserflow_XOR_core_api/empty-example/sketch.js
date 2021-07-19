//https://www.tensorflow.org/js/guide/train_models

//MODEL (params são os tamanhos dos arrays/matrizes)
const w1 = tf.variable(tf.randomUniform([2, 2])); //inputs (primeiros nós) , numero de nós da layer
const b1 = tf.variable(tf.randomNormal([2])); //uma matriz com 2 randoms de -1 a 1 -> p.e. [0.4, -0.5], ou seja, uma random por cada nó
const w2 = tf.variable(tf.randomUniform([2, 1])); //inputs (nós anterios), outputs (ultimo nó)
const b2 = tf.variable(tf.randomNormal([1])); //
const model = function (x) {
  return x.matMul(w1).add(b1).sigmoid().matMul(w2).add(b2).sigmoid();
}

//OPTIMIZER
//const optimizer = tf.train.sgd(0.5);
const optimizer = tf.train.adam(0.1);

//TRAINING DATA
const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]); //tensor 2d porque tem varios arrays 1d la dentro
const ys = tf.tensor2d([[0], [1], [1], [0]]); //tensor 2d porque tem varios arrays 1d la dentro

//INPUTS I WANT TO PREDICT
let txs = [];

//DESENHAR
let res = 20, rw = 400 / res, fills = [], waiting = false;

function setup() {
  createCanvas(400, 400);

  let i = 0;
  for (let x = 0; x < res; x++) for (let y = 0; y < res; y++) {
    let xv = map(x, 0, res - 1, 0, 1), yv = map(y, 0, res - 1, 0, 1);
    txs[i] = [xv, yv]; //iniciar inputs que quero prever
    fills[i] = 0; //iniciar os fills a zero (preto)
    i++;
  }

  txs = tf.tensor2d(txs); //converter inputs em tensor
}

//CRIAR DATA
function* data() {
  let d = xs.dataSync();
  for (let i = 0; i < d.length; i += 2) {
    yield [d[i], d[i + 1]];
  }
}

function* labels() {
  let d = ys.dataSync();
  for (let i = 0; i < d.length; i++) {
    yield [d[i]];
  }
}

const trainingXs = tf.data.generator(data);
const trainingYs = tf.data.generator(labels);
const ds = tf.data.zip({trainingXs, trainingYs}) //criar dataset zippado
  .shuffle(4/* bufferSize */) //desordenar o dataset de forma semi-aleatoria
  .batch(32); //separar o dataset em grupos de 32 linhas

async function train() {
  for (let epoch = 0; epoch < 50; epoch++) {
    await ds.forEachAsync(({trainingXs, trainingYs}) => { //por cada grupo de 32 linhas do dataset -> retornar trainingXs e trainingYs
      optimizer.minimize(() => { //minimizar o erro no sgd/adam. minimize() recebe esta funcao que retorna o erro
        const predYs = model(trainingXs);
        return tf.losses.meanSquaredError(trainingYs, predYs); //.data().then(l => console.log('Loss', l));
      });
    });
  }
}

function draw() {
  background(0);

  //TREINAR
  if (!waiting) {
    waiting = true;
    train().then(() => {
      waiting = false;

      //PREVER
      fills = model(txs).dataSync();
    });
  }

  //DESENHAR PREVISOES
  let i = 0;
  for (let x = 0; x < res; x++)
    for (let y = 0; y < res; y++) {
      fill(fills[i] * 255);
      stroke(255);
      rect(x * rw, y * rw, rw, rw);
      i++;
    }

}



