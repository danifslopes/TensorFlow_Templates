let model, res = 20, rw = 400 / res, fills = [], txs = [], waiting = false;

const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]); //tensor 2d porque tem varios arrays 1d la dentro
const ys = tf.tensor2d([[0], [1], [1], [0]]); //tensor 2d porque tem varios arrays 1d la dentro

function setup() {
  createCanvas(400, 400);
  model = tf.sequential();
  let hidden = tf.layers.dense({inputShape: [2], units: 2, activation: 'sigmoid'});
  let output = tf.layers.dense({units: 1, activation: 'sigmoid'});
  model.add(hidden);
  model.add(output);
  model.compile({optimizer: tf.train.adam(0.1), loss: 'meanSquaredError'});
  //model.compile({optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError'});
  let i = 0;
  for (let x = 0; x < res; x++) for (let y = 0; y < res; y++) {
    let xv = map(x, 0, res - 1, 0, 1), yv = map(y, 0, res - 1, 0, 1);
    txs[i] = [xv, yv];
    fills[i] = 0;
    i++;
  }
  txs = tf.tensor2d(txs);
}

function draw() {
  background(0);

  if (!waiting) {
    waiting = true;

    model.fit(xs, ys, {shuffle: true, epochs: 200}).then((response) => { //traiin
      console.log(response.history.loss[0]);

      fills = model.predict(txs).dataSync(); //predict
      waiting = false;
    });
  }

  let i = 0;
  for (let x = 0; x < res; x++)
    for (let y = 0; y < res; y++) {
      fill(fills[i] * 255);
      stroke(255);
      rect(x * rw, y * rw, rw, rw);
      i++;
    }

}
