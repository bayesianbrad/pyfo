(let [x1 (sample (normal 0 1))
      x2 (sample (normal 0 1))]
  (if (> x1 0)
    (observe (normal x2 1) 1)
    (observe (normal -1 1) 1))
  x1)