(let [x1 (sample (normal 0 1))
      x2 (sample (normal 1 2))
      x3 (+ x1 x2)]
  (if (> x3 0)
    (observe (normal 1 1) 1)
    (observe (normal -1 1) 1)
;  (let [x3 (sample (normal 0 5))
;        b  (- x1 x3)]
;    (if (> b a)
;      (observe (gamma 2 4) 2)
;      (observe (beta  3 1)3))
  x1 x2 x3))