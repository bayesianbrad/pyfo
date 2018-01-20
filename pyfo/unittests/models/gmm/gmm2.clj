(let [mu1 (sample (normal -5 1))
      mu2 (sample (normal 5 1))
      sig1 (sample (poisson 2))
      mu (vector mu1 mu2)
      z (sample (categorical [0.1 0.9]))
      y 3]
  (observe (normal (get mu z) sig1) y)
  (vector z mu ))