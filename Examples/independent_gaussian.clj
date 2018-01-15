(let [x1      (sample (normal 1.0 5.0))
      x2      (sample (normal 2.0 3.0))
      x3      (sample (gamma 3.0 4.0))]
  [x1 x2 x3])