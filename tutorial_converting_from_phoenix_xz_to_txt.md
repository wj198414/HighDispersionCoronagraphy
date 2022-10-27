```
xz -d lte028.0-5.5-0.0a+0.0.BT-Settl.spec.7.xz
cat lte028.0-5.5-0.0a+0.0.BT-Settl.spec.7 | awk '{print $1,$2}' | sed s/D/E/g> lte028.0-5.5-0.0a+0.0.BT-Settl.spec.txt
```

[10**(F_lam + DF) to convert to Ergs/sec/cm**2/A](https://phoenix.ens-lyon.fr/Grids/FORMAT)
