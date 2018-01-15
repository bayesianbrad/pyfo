;BRADLEY : THIS IS NOT A VALID MODEL!
(let [x (sample (normal 0 1))]
  (if (> x 0)
    (if (< x 1)
      (observe (normal 0.5 1) 1)
      (observe (normal 2 1) 1))
  (if (> x -1)
    (observe (normal -0.5 1) 1)
    (observe (normal -2 1) 1)))
  x)