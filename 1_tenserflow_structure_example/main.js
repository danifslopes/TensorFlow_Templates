const model = tf.sequential(),

  hidden = tf.layers.dense({
    inputShape: [2], //numero de inputs (1d com 2 valores). especificado apenas na primeira layer
    units: 4, //numero de neuronios da hidden layer
    activation: 'sigmoid',
  }),

  output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  });

model.add(hidden);
model.add(output);
model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: tf.losses.meanSquaredError
});

//train
//inputs
const xs = tf.tensor2d([ //tensor 2d porque tem varios arrays 1d la dentro
  [0, 0],
  [0.5, 0.5],
  [1, 1]
]);

//outputs
const ys = tf.tensor2d([ //tensor 2d porque tem varios arrays 1d la dentro
  [1],
  [0.5],
  [0]
]);

//model.fit(xs, ys, {shuffle: true, epochs: 1000}).then(response => {
//  console.log(response);
//  console.log(response.history.loss[0]);
//});

async function train() {
  const response = await model.fit(xs, ys, {shuffle: true, epochs: 1000});
  console.log(response.history.loss[0]);
}

train().then(() => {
  console.log("Train finished");
  let outputs = model.predict(xs);
  outputs.print();

});




