;(let [mu (vector -5 5)
;      obs (vector -7 7)
;      z (sample (categorical [0.3 0.7]))]
;  (observe (normal (get mu z) 2) (get obs z))
;  (vector z (get mu z)))

(let [mu (vector -5 5)
      obs 7
      z (sample (categorical [0.5 0.5]))]
  (observe (normal (get mu z) 2) obs)
  (vector z mu))