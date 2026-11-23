import model from './model.json'

// Runs the random forest exported by training/export_model.py.
// Matches scikit-learn: average each tree's leaf probabilities, pick the highest.
export const CLASSES = model.classes

function leafFor(tree, x) {
  let node = 0
  while (tree.left[node] !== -1) {
    node = x[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node]
  }
  return tree.value[node]
}

// `landmarks` is the 21 {x, y, z} points from the hand landmarker.
export function predictSign(landmarks) {
  // scikit-learn compares in 32-bit floats, so round the inputs the same way.
  const x = landmarks.flatMap((p) => [p.x, p.y, p.z]).map(Math.fround)
  if (x.length !== model.nFeatures) throw new Error(`Expected ${model.nFeatures} values, got ${x.length}`)

  const scores = new Array(CLASSES.length).fill(0)
  for (const tree of model.trees) {
    const leaf = leafFor(tree, x)
    for (let i = 0; i < scores.length; i++) scores[i] += leaf[i]
  }
  let best = 0
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i
  return { label: CLASSES[best], confidence: scores[best] / model.trees.length }
}
