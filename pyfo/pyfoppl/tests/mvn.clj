(let [x (sample (mvn [0 0] [[1 0] [0 1]]))
      y [7 7]]
  (observe (mvn x [[2 0] [0 2]]) y)
  x)