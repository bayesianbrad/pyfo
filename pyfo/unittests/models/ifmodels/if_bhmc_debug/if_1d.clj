(let [x (sample (normal 0 1))]
  (if (> x 0)
    (observe (normal 1 1) 1)
    (observe (normal -1 1) 1))
  x)

;(let [x1 (sample (normal 0 1))
;      x2 (sample (normal 1 2))
;      x3 (+ x1 x2)]
;  (if (> x3 0)
;    (observe (normal 1 1) 1)
;    (if (> x3 2)
;      (observe (normal 1 3)))
;    (observe (normal -1 1) 1)
;;  (let [x3 (sample (normal 0 5))
;;        b  (- x1 x3)]
;;    (if (> b a)
;;      (observe (gamma 2 4) 2)
;;      (observe (beta  3 1)3))
;  x1 x2 x3))
;(let [x1 (sample (normal 0 1))
;      x2 (sample (normal 0 1))
;      cov [[1 0] [0 1]]
;      y [1 1]]
;  (if (> x1 0)
;    (if (> x2 0)
;      (observe (mvn [1 1] cov) y)
;      (observe (mvn [1 -1] cov) y))
;    (if (> x2 0)
;      (observe (mvn [-1 1] cov) y)
;      (observe (mvn [-1 -1] cov) y)))
;  [x1 x2])
