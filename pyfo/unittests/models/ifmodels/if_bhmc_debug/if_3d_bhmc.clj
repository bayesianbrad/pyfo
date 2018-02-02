(let [x1 (sample (normal 0 2))  ;if
      x2 (sample (normal 0 2))  ;cont
      x3 (sample (categorical [0.5 0.5]))  ;disc
      y 10]
  (if (> x1 0)
    (observe (normal x2 1) y)
    (observe (normal x3 1) y))
  (vector x1 x2 x3))