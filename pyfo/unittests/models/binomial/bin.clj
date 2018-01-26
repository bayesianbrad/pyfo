(let [x2 (sample (binomial  1 [0.1]))
      x1 (sample (binomial  2 [0.5]))]
  x2 x1)