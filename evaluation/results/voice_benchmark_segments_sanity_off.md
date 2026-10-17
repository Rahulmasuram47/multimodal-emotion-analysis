# Voice Backend Benchmark (Mode: segments, Sanity: off)

## superb

| File | True | Raw pred / raw conf | Final pred / final conf |
|------|------|---------------------|-------------------------|
| Happy2.wav | happy | happy / 0.943 | happy / 0.943 |
| happy_sample.wav | happy | sad / 0.738 | sad / 0.738 |
| angry_sample.wav | angry | sad / 0.674 | sad / 0.674 |
| Neutral1.wav | neutral | angry / 0.613 | angry / 0.613 |
| Neutral2.wav | neutral | angry / 0.602 | angry / 0.602 |
| Sad1.wav | sad | happy / 0.615 | happy / 0.615 |
| Sad2.wav | sad | neutral / 0.381 | neutral / 0.381 |
| Sad3.wav | sad | happy / 0.901 | happy / 0.901 |

**N:** 8  
**Accuracy:** 12.5% (95% Wilson CI: 2.2% - 47.1%)  
**Macro-F1:** 0.100 (4 classes; other=wrong)  
**Mean conf correct:** 0.943  
**Mean conf wrong:** 0.646  
**Mean s/clip:** 9.456  
**Predicted-class share:** angry: 25.0%, happy: 37.5%, neutral: 12.5%, sad: 25.0%  

**Confusion matrix** (rows=true, cols=pred):

```
        neut    happ     sad    angr
neutra     0       0       0       2
happy      0       1       1       0
sad        1       2       0       0
angry      0       0       1       0
```

## emotion2vec

| File | True | Raw pred / raw conf | Final pred / final conf |
|------|------|---------------------|-------------------------|
| Happy2.wav | happy | happy / 1.000 | happy / 1.000 |
| happy_sample.wav | happy | neutral / 1.000 | neutral / 1.000 |
| angry_sample.wav | angry | neutral / 0.999 | neutral / 0.999 |
| Neutral1.wav | neutral | neutral / 1.000 | neutral / 1.000 |
| Neutral2.wav | neutral | neutral / 1.000 | neutral / 1.000 |
| Sad1.wav | sad | neutral / 1.000 | neutral / 1.000 |
| Sad2.wav | sad | sad / 0.932 | sad / 0.932 |
| Sad3.wav | sad | sad / 1.000 | sad / 1.000 |

**N:** 8  
**Accuracy:** 62.5% (95% Wilson CI: 30.6% - 86.3%)  
**Macro-F1:** 0.510 (4 classes; other=wrong)  
**Mean conf correct:** 0.986  
**Mean conf wrong:** 1.000  
**Mean s/clip:** 15.311  
**Predicted-class share:** happy: 12.5%, neutral: 62.5%, sad: 25.0%  

**Confusion matrix** (rows=true, cols=pred):

```
        neut    happ     sad    angr
neutra     2       0       0       0
happy      1       1       0       0
sad        1       0       2       0
angry      1       0       0       0
```

