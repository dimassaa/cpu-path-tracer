# Improvements

Tracked, not yet implemented. Ordered roughly by payoff.

- **Multiple importance sampling.** NEE and BSDF sampling are combined by a
  hard rule (emission counted only after specular bounces). MIS with the power
  heuristic would remove that special case and reduce variance on the fuzzy
  metal.
- **Environment importance sampling.** The sky gradient is only found by BSDF
  sampling, so it contributes little and noisily. A small 2D CDF over the
  gradient would light the open scene much faster.
- **BVH.** Intersection is linear in the primitive count. Fine for the current
  scenes (a few dozen), unusable beyond a few hundred.
- **Adaptive sampling.** The noise level is very uneven (the ceiling gets no
  direct light). A per-tile variance estimate would spend samples where they
  matter and cut render time several-fold at equal quality.
- **Checker texture filtering.** The distance fade hides aliasing but is a
  hack; proper mip selection from ray differentials would be correct.
- **Colour management.** Gamma 2.2 instead of the piecewise sRGB transfer
  function, and 8-bit output only. EXR output plus a real tonemap would make
  the renderer usable for compositing.
- **Denoising.** A simple non-local-means or guided filter would let the
  Cornell box run at a fraction of the current spp.
- **Bloom.** The light panel is clipped flat white; a small post-process bloom
  would sell the brightness better than a hard clip.
- **Multiple lights.** `rect_light` is a single global; moving to an array is
  mechanical but would make the code less tidy.
- **Relative epsilons.** Shadow-ray origins and `tmin` use absolute values
  (1e-3, 1e-4). They are tuned for scenes hundreds of units across; at metre
  scale or at kilometre scale they will either leak light or reintroduce
  self-shadowing. Deriving them from the scene scale is the proper fix.
- **Seed control.** Renders are deterministic per pixel but there is no CLI
  seed, so two images at the same spp cannot be averaged into one.
