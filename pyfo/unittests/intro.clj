(let [paper_Accepted (sample (binomial 1 [0.25]))
      alice_writes_good_paper (sample (normal 2 4))
      alice_goes 1
      alice_stays 0]
      (if (> alice_writes_good_paper 2)
        (observe (normal alice_writes_good_paper 1) alice_goes)
        (observe (normal alice_writes_good_paper 7) alice_stays)
        )
  alice_writes_good_paper)
