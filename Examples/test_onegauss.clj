(let [x1      (sample (normal 1.0 5.0))
      x2 (+ x1)]
  (observe (normal x2 2.0) 7.0)
  [x1])