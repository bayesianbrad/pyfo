(let [x1 (sample (normal 0 1))
      y 10]
  (if (> x1 0)
    (let [x2 (sample (normal 0 1))]
      (observe (normal x2 2) y))
    (let [x3 (sample (normal 0 1))]
      (observe (normal x3 2) y)))
  x1)

