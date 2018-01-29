
(let [x1 (sample (normal 0 1))
      x2 (sample (normal 0 1))
      x3 (sample (normal 0 1))
      y 1]
  (if (> x1 0)
    (observe (normal x2 2) y)
    (observe (normal x3 2) y))
  (vector x1 x2 x3))