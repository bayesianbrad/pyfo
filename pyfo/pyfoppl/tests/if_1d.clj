(let [x (sample (normal 0 1))]
  (if (> x 0)
    (observe (normal 1 1) 1)
    (observe (normal -1 1) 1))
  x)