(let [score (sample (normal 1 4))
      alice_goes 1]
      (if (> score 2)
        (observe (binomial 1 [0.9]) alice_goes)
        (observe (binomial 1 [0.3]) alice_goes)
        )
  score)
