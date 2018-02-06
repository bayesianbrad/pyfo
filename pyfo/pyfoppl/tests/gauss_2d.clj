(let [x1 (sample (normal 1.0 5.0))
      x2 (sample (normal 1.0 5.0))
      y 7.0]
  (observe (normal x1 2.0) y)
  (observe (normal x2 2.0) y)
  [x1 x2])