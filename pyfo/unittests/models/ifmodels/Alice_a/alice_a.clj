(let [s (sample (normal 0 4))
       g 1]
   (if (> s 0)
;        (observe (bernoulli 0.9) g)
;        (observe (bernoulli 0.3) g))
        (observe (binomial 1 [0.9]) g)
        (observe (binomial 1 [0.3]) g)
     s))