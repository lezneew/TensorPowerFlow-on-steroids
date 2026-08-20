# R/X-Abschnitt: Abbildungen, Tabellen, Kennzahlen

Zeilen: 780 | PQ: 78 | PV: 702 | PV gekoppelt: 234

## Klassifikation der PV-Laeufe (alle Varianten)

```
           cls  faelle
0    Divergenz     481
1         conv     217
2  Grenzzyklus       4
```

## Soll-Ist gegen den Text

```
                                      key       text      daten     delta     tol      status
0          inner.eta_rise_pct.const_z.n40    2.20000    2.20000   0.00000  0.3000          ok
1         inner.eta_rise_pct.const_z.n120    5.80000    5.80000   0.00000  0.3000          ok
2         inner.eta_rise_pct.const_z.n350   21.70000   21.70000   0.00000  0.5000          ok
3              inner.vmin_lo.const_z.n350    0.97250    0.97250   0.00000  0.0010          ok
4              inner.vmin_hi.const_z.n350    0.87740    0.87740   0.00000  0.0010          ok
5          inner.eta_pred_hi.const_z.n350    0.13430    0.13432   0.00002  0.0005          ok
6               inner.eta_hi.const_z.n350    0.13306    0.13306   0.00000  0.0005          ok
7         inner.pred_err_pct.const_z.n350    0.90000    0.94000   0.04000  0.2000          ok
8            inner.eta_growth.const_x.n40   13.80000   13.81000   0.01000  0.2000          ok
9                  inner.k_hi.const_x.n40   16.00000   16.00000   0.00000  0.0000          ok
10          inner.vmin_factor.const_x.n40    1.43600    1.43600   0.00000  0.0100          ok
11                 inner.crossing.dev_pct    0.20000        NaN       NaN  0.1000       fehlt
12                  inner.eta_max_success    0.62000    0.62600   0.00600  0.0200          ok
13                         kappa.mean.n40    0.01260    0.01290   0.00030  0.0002  ABWEICHUNG
14                        kappa.mean.n120    0.03260    0.03310   0.00050  0.0005          ok
15                        kappa.mean.n350    0.10340    0.10260  -0.00080  0.0010          ok
16                   kappa.spread_pct_max    3.40000    7.70000   4.30000  0.2000  ABWEICHUNG
17                    kappa.exponent_in_n    0.97000    0.95600  -0.01400  0.0200          ok
18                   xpp.cond_var_pct_max    0.00000    0.00000   0.00000  0.0500          ok
19                xpp.cond_growth_npv.n40   21.70000   21.52000  -0.18000  0.3000          ok
20               xpp.cond_growth_npv.n120   28.60000   28.71000   0.11000  0.3000          ok
21               xpp.cond_growth_npv.n350   21.60000   21.58000  -0.02000  0.3000          ok
22                  outer.q_growth.pv0.10   20.20000   20.20000   0.00000  0.5000          ok
23                  outer.q_growth.pv0.25   30.10000   30.10000   0.00000  0.5000          ok
24                  outer.q_growth.pv0.50   18.70000   18.70000   0.00000  0.5000          ok
25                      outer.sens_corr_q    0.67000    0.75000   0.08000  0.0300  ABWEICHUNG
26         outer.inner_per_outer_conv_med    5.50000    4.50000  -1.00000  0.2000  ABWEICHUNG
27                outer.k_max_below_limit    7.00000   17.00000  10.00000  0.0000  ABWEICHUNG
28            outer.quote.const_z.coupled   88.90000   88.90000   0.00000  0.3000          ok
29            outer.quote.const_x.coupled   74.40000   74.40000   0.00000  0.3000          ok
30          outer.quote.const_z.decoupled   12.80000   12.80000   0.00000  0.3000          ok
31     outer.rho_star_med.const_z.coupled    7.32000    7.32000   0.00000  0.0500          ok
32     outer.rho_star_med.const_x.coupled    4.64000    4.64000   0.00000  0.0500          ok
33        outer.k_out_med.const_z.coupled    3.00000    3.00000   0.00000  0.0000          ok
34         outer.k_in_med.const_z.coupled   16.50000   16.50000   0.00000  0.5000          ok
35  pred.spearman_outer.sens_error_median    0.89900    0.89900   0.00000  0.0100          ok
36                 pred.spearman_outer.rx    0.53700    0.53700   0.00000  0.0100          ok
37         pred.spearman_outer.rho_jacobi   -0.24000   -0.24000   0.00000  0.0100          ok
38              pred.spearman_outer.nodes   -0.06800   -0.06800   0.00000  0.0100          ok
39                            pred.n_runs  217.00000  217.00000   0.00000  0.0000          ok
40                        nr.faelle.beide  191.00000  191.00000   0.00000  0.0000          ok
41                       nr.faelle.keines   41.00000   41.00000   0.00000  0.0000          ok
42                       nr.faelle.nur NR    2.00000    2.00000   0.00000  0.0000          ok
43                      nr.faelle.nur TPF    0.00000    0.00000   0.00000  0.0000          ok
44                              nr.n_runs  234.00000  234.00000   0.00000  0.0000          ok
45                   nr.skipped_in_keines   15.00000    0.00000 -15.00000  0.0000  ABWEICHUNG
```

## Kennzahlen

```
                                              key              value  unit                                              note
0                    inner.eta_fit_vs_emp_max_pct              0.746     %          Konsistenz Log-Fit gegen empirische Rate
1                                inner.fit_r2_min           0.148466                    kleinstes R^2 des geometrischen Fits
2                              inner.crossing.rho              0.147                               |z| in beiden Modi gleich
3                        inner.eta_lo.const_x.n40            0.01292                                                        
4                        inner.eta_hi.const_x.n40            0.17845                                              bei rho=10
5                    inner.eta_growth.const_x.n40              13.81                                                        
6                  inner.eta_rise_pct.const_x.n40             1281.0     %                                                  
7                       inner.vmin_lo.const_x.n40             0.9971  p.u.                                                  
8                       inner.vmin_hi.const_x.n40              0.832  p.u.                                                  
9                          inner.k_lo.const_x.n40                7.0                                                        
10                         inner.k_hi.const_x.n40               16.0                                                        
11                  inner.eta_pred_hi.const_x.n40             0.1856        eta_lo * z_rel-Verhaeltnis / v_min^2-Verhaeltnis
12                 inner.pred_err_pct.const_x.n40               4.01     %                                                  
13                  inner.vmin_factor.const_x.n40              1.436                                                        
14                    inner.k_pred_lo.const_x.n40                6.4                                         ln(tol)/ln(eta)
15                    inner.k_pred_hi.const_x.n40               16.0                                         ln(tol)/ln(eta)
16                       inner.eta_lo.const_z.n40            0.01302                                                        
17                       inner.eta_hi.const_z.n40             0.0133                                              bei rho=10
18                   inner.eta_growth.const_z.n40               1.02                                                        
19                 inner.eta_rise_pct.const_z.n40                2.2     %                                                  
20                      inner.vmin_lo.const_z.n40             0.9971  p.u.                                                  
21                      inner.vmin_hi.const_z.n40             0.9855  p.u.                                                  
22                         inner.k_lo.const_z.n40                7.0                                                        
23                         inner.k_hi.const_z.n40                7.0                                                        
24                  inner.eta_pred_hi.const_z.n40            0.01333        eta_lo * z_rel-Verhaeltnis / v_min^2-Verhaeltnis
25                 inner.pred_err_pct.const_z.n40                0.2     %                                                  
26                  inner.vmin_factor.const_z.n40              1.024                                                        
27                    inner.k_pred_lo.const_z.n40                6.4                                         ln(tol)/ln(eta)
28                    inner.k_pred_hi.const_z.n40                6.4                                         ln(tol)/ln(eta)
29                      inner.eta_lo.const_x.n120             0.0336                                                        
30                      inner.eta_hi.const_x.n120            0.53432                                          bei rho=6.8129
31                  inner.eta_growth.const_x.n120               15.9                                                        
32                inner.eta_rise_pct.const_x.n120             1490.1     %                                                  
33                     inner.vmin_lo.const_x.n120             0.9924  p.u.                                                  
34                     inner.vmin_hi.const_x.n120             0.6231  p.u.                                                  
35                        inner.k_lo.const_x.n120                9.0                                                        
36                        inner.k_hi.const_x.n120               42.0                                                        
37                 inner.eta_pred_hi.const_x.n120            0.58391        eta_lo * z_rel-Verhaeltnis / v_min^2-Verhaeltnis
38                inner.pred_err_pct.const_x.n120               9.28     %                                                  
39                 inner.vmin_factor.const_x.n120              2.536                                                        
40                   inner.k_pred_lo.const_x.n120                8.1                                         ln(tol)/ln(eta)
41                   inner.k_pred_hi.const_x.n120               44.1                                         ln(tol)/ln(eta)
42                      inner.eta_lo.const_z.n120            0.03387                                                        
43                      inner.eta_hi.const_z.n120            0.03583                                              bei rho=10
44                  inner.eta_growth.const_z.n120               1.06                                                        
45                inner.eta_rise_pct.const_z.n120                5.8     %                                                  
46                     inner.vmin_lo.const_z.n120             0.9923  p.u.                                                  
47                     inner.vmin_hi.const_z.n120             0.9629  p.u.                                                  
48                        inner.k_lo.const_z.n120                9.0                                                        
49                        inner.k_hi.const_z.n120                9.0                                                        
50                 inner.eta_pred_hi.const_z.n120            0.03597        eta_lo * z_rel-Verhaeltnis / v_min^2-Verhaeltnis
51                inner.pred_err_pct.const_z.n120                0.4     %                                                  
52                 inner.vmin_factor.const_z.n120              1.062                                                        
53                   inner.k_pred_lo.const_z.n120                8.2                                         ln(tol)/ln(eta)
54                   inner.k_pred_hi.const_z.n120                8.3                                         ln(tol)/ln(eta)
55                      inner.eta_lo.const_x.n350            0.10843                                                        
56                      inner.eta_hi.const_x.n350            0.62632                                          bei rho=2.1544
57                  inner.eta_growth.const_x.n350               5.78                                                        
58                inner.eta_rise_pct.const_x.n350              477.6     %                                                  
59                     inner.vmin_lo.const_x.n350             0.9727  p.u.                                                  
60                     inner.vmin_hi.const_x.n350             0.6045  p.u.                                                  
61                        inner.k_lo.const_x.n350               13.0                                                        
62                        inner.k_hi.const_x.n350               55.0                                                        
63                 inner.eta_pred_hi.const_x.n350            0.66364        eta_lo * z_rel-Verhaeltnis / v_min^2-Verhaeltnis
64                inner.pred_err_pct.const_x.n350               5.96     %                                                  
65                 inner.vmin_factor.const_x.n350               2.59                                                        
66                   inner.k_pred_lo.const_x.n350               12.4                                         ln(tol)/ln(eta)
67                   inner.k_pred_hi.const_x.n350               59.1                                         ln(tol)/ln(eta)
68                      inner.eta_lo.const_z.n350            0.10934                                                        
69                      inner.eta_hi.const_z.n350            0.13306                                              bei rho=10
70                  inner.eta_growth.const_z.n350               1.22                                                        
71                inner.eta_rise_pct.const_z.n350               21.7     %                                                  
72                     inner.vmin_lo.const_z.n350             0.9725  p.u.                                                  
73                     inner.vmin_hi.const_z.n350             0.8774  p.u.                                                  
74                        inner.k_lo.const_z.n350               13.0                                                        
75                        inner.k_hi.const_z.n350               14.0                                                        
76                 inner.eta_pred_hi.const_z.n350            0.13432        eta_lo * z_rel-Verhaeltnis / v_min^2-Verhaeltnis
77                inner.pred_err_pct.const_z.n350               0.94     %                                                  
78                 inner.vmin_factor.const_z.n350              1.228                                                        
79                   inner.k_pred_lo.const_z.n350               12.5                                         ln(tol)/ln(eta)
80                   inner.k_pred_hi.const_z.n350               13.7                                         ln(tol)/ln(eta)
81                          inner.eta_max_success              0.626                                                        
82         inner.nonconv.eta_1.const_x.n120.rho10              3.732                                                        
83         inner.nonconv.eta_2.const_x.n120.rho10              1.928                                                        
84       inner.nonconv.eta_inf.const_x.n120.rho10              1.908                                                        
85         inner.nonconv.v_min.const_x.n120.rho10              0.432                                                        
86     inner.nonconv.eta_1.const_x.n350.rho3.1623              5.284                                                        
87     inner.nonconv.eta_2.const_x.n350.rho3.1623               2.28                                                        
88   inner.nonconv.eta_inf.const_x.n350.rho3.1623               2.16                                                        
89     inner.nonconv.v_min.const_x.n350.rho3.1623              0.405                                                        
90     inner.nonconv.eta_1.const_x.n350.rho4.6416              4.157                                                        
91     inner.nonconv.eta_2.const_x.n350.rho4.6416              1.794                                                        
92   inner.nonconv.eta_inf.const_x.n350.rho4.6416              1.699                                                        
93     inner.nonconv.v_min.const_x.n350.rho4.6416              0.546                                                        
94     inner.nonconv.eta_1.const_x.n350.rho6.8129              5.896                                                        
95     inner.nonconv.eta_2.const_x.n350.rho6.8129              2.544                                                        
96   inner.nonconv.eta_inf.const_x.n350.rho6.8129              2.411                                                        
97     inner.nonconv.v_min.const_x.n350.rho6.8129              0.552                                                        
98         inner.nonconv.eta_1.const_x.n350.rho10              3.701                                                        
99         inner.nonconv.eta_2.const_x.n350.rho10              1.597                                                        
100      inner.nonconv.eta_inf.const_x.n350.rho10              1.513                                                        
101        inner.nonconv.v_min.const_x.n350.rho10              0.842                                                        
102                                kappa.mean.n40             0.0129                                                        
103                          kappa.spread_pct.n40                3.4     %                                                  
104                              kappa.points.n40                 26                                                        
105                               kappa.mean.n120             0.0331                                                        
106                         kappa.spread_pct.n120                7.7     %                                                  
107                             kappa.points.n120                 25                                                        
108                               kappa.mean.n350             0.1026                                                        
109                         kappa.spread_pct.n350                4.9     %                                                  
110                             kappa.points.n350                 22                                                        
111                          kappa.spread_pct_max                7.7     %                                                  
112                           kappa.exponent_in_n              0.956                                log-log-Fit kappa_bar(n)
113                                  kappa.kappa1           0.000293             kappa_bar/n, Vorfaktor in Gl. eta-empirisch
114            kappa.from_eta2.n120.const_x.rho10              0.036                        Einordnung des divergenten Laufs
115        kappa.from_eta2.n350.const_x.rho3.1623              0.114                        Einordnung des divergenten Laufs
116        kappa.from_eta2.n350.const_x.rho4.6416              0.114                        Einordnung des divergenten Laufs
117        kappa.from_eta2.n350.const_x.rho6.8129              0.114                        Einordnung des divergenten Laufs
118            kappa.from_eta2.n350.const_x.rho10              0.114                        Einordnung des divergenten Laufs
119                         outer.q_growth.pv0.10               20.2                                                        
120                       outer.sens_slope.pv0.10               1.21                           log-log-Steigung eps_lin(rho)
121                         outer.q_growth.pv0.25               30.1                                                        
122                       outer.sens_slope.pv0.25               1.19                           log-log-Steigung eps_lin(rho)
123                         outer.q_growth.pv0.50               18.7                                                        
124                       outer.sens_slope.pv0.50               1.19                           log-log-Steigung eps_lin(rho)
125                          outer.lever_expected               10.0                           sqrt(1+rho^2) ueber den Sweep
126                             outer.sens_corr_q               0.75                                                        
127                outer.inner_per_outer_conv_med                4.5                                                        
128               outer.inner_per_outer.Divergenz        114.6-196.8                                                        
129             outer.inner_per_outer.Grenzzyklus            5.0-5.2                                                        
130                       outer.k_max_below_limit                 17                            max. k_out fuer eps_lin<=0.6
131                  outer.ms_per_inner_slope.n40              -0.12                                                        
132                 outer.ms_per_inner_slope.n120              -0.18                                                        
133                 outer.ms_per_inner_slope.n350              -0.84                                                        
134                   outer.quote.const_x.coupled               74.4     %                                        117 Laeufe
135               outer.k_out_med.const_x.coupled                3.0                                                        
136                outer.k_in_med.const_x.coupled               16.0                                                        
137            outer.rho_star_med.const_x.coupled               4.64                                                        
138                 outer.quote.const_x.decoupled                9.4     %                                        117 Laeufe
139             outer.k_out_med.const_x.decoupled               32.0                                                        
140              outer.k_in_med.const_x.decoupled              115.0                                                        
141          outer.rho_star_med.const_x.decoupled                0.1                                                        
142                     outer.quote.const_x.exact                0.0     %                                        117 Laeufe
143              outer.rho_star_med.const_x.exact                0.1                                                        
144                   outer.quote.const_z.coupled               88.9     %                                        117 Laeufe
145               outer.k_out_med.const_z.coupled                3.0                                                        
146                outer.k_in_med.const_z.coupled               16.5                                                        
147            outer.rho_star_med.const_z.coupled               7.32                                                        
148                 outer.quote.const_z.decoupled               12.8     %                                        117 Laeufe
149             outer.k_out_med.const_z.decoupled               31.0                                                        
150              outer.k_in_med.const_z.decoupled              114.0                                                        
151          outer.rho_star_med.const_z.decoupled                0.1                                                        
152                     outer.quote.const_z.exact                0.0     %                                        117 Laeufe
153              outer.rho_star_med.const_z.exact                0.1                                                        
154               xpp.cond_xpp_var_pct.n40.pv0.10                0.0     %                               Variation ueber rho
155             xpp.rho_jacobi_var_pct.n40.pv0.10                0.0     %                               Variation ueber rho
156           xpp.diag_dom_min_var_pct.n40.pv0.10                0.0     %                               Variation ueber rho
157               xpp.cond_xpp_var_pct.n40.pv0.25                0.0     %                               Variation ueber rho
158             xpp.rho_jacobi_var_pct.n40.pv0.25                0.0     %                               Variation ueber rho
159           xpp.diag_dom_min_var_pct.n40.pv0.25                0.0     %                               Variation ueber rho
160               xpp.cond_xpp_var_pct.n40.pv0.50                0.0     %                               Variation ueber rho
161             xpp.rho_jacobi_var_pct.n40.pv0.50                0.0     %                               Variation ueber rho
162           xpp.diag_dom_min_var_pct.n40.pv0.50                0.0     %                               Variation ueber rho
163              xpp.cond_xpp_var_pct.n120.pv0.10                0.0     %                               Variation ueber rho
164            xpp.rho_jacobi_var_pct.n120.pv0.10                0.0     %                               Variation ueber rho
165          xpp.diag_dom_min_var_pct.n120.pv0.10                0.0     %                               Variation ueber rho
166              xpp.cond_xpp_var_pct.n120.pv0.25                0.0     %                               Variation ueber rho
167            xpp.rho_jacobi_var_pct.n120.pv0.25                0.0     %                               Variation ueber rho
168          xpp.diag_dom_min_var_pct.n120.pv0.25                0.0     %                               Variation ueber rho
169              xpp.cond_xpp_var_pct.n120.pv0.50                0.0     %                               Variation ueber rho
170            xpp.rho_jacobi_var_pct.n120.pv0.50                0.0     %                               Variation ueber rho
171          xpp.diag_dom_min_var_pct.n120.pv0.50                0.0     %                               Variation ueber rho
172              xpp.cond_xpp_var_pct.n350.pv0.10                0.0     %                               Variation ueber rho
173            xpp.rho_jacobi_var_pct.n350.pv0.10                0.0     %                               Variation ueber rho
174          xpp.diag_dom_min_var_pct.n350.pv0.10                0.0     %                               Variation ueber rho
175              xpp.cond_xpp_var_pct.n350.pv0.25                0.0     %                               Variation ueber rho
176            xpp.rho_jacobi_var_pct.n350.pv0.25                0.0     %                               Variation ueber rho
177          xpp.diag_dom_min_var_pct.n350.pv0.25                0.0     %                               Variation ueber rho
178              xpp.cond_xpp_var_pct.n350.pv0.50                0.0     %                               Variation ueber rho
179            xpp.rho_jacobi_var_pct.n350.pv0.50                0.0     %                               Variation ueber rho
180          xpp.diag_dom_min_var_pct.n350.pv0.50                0.0     %                               Variation ueber rho
181                          xpp.cond_var_pct_max                0.0     %       groesste Variation von cond(X_pp) ueber rho
182                       xpp.cond_growth_npv.n40              21.52                                              n_pv 4->20
183                     xpp.first_pv_rhoJ_gt1.n40               0.25                         kleinster PV-Anteil mit rho_J>1
184                      xpp.cond_growth_npv.n120              28.71                                             n_pv 12->60
185                    xpp.first_pv_rhoJ_gt1.n120                0.1                         kleinster PV-Anteil mit rho_J>1
186                      xpp.cond_growth_npv.n350              21.58                                            n_pv 35->175
187                    xpp.first_pv_rhoJ_gt1.n350                0.1                         kleinster PV-Anteil mit rho_J>1
188                        outer.plateau.n40.npv4                  3                                                        
189               outer.rho_star.const_z.n40.npv4               10.0                                             Grenzzyklus
190                       outer.plateau.n40.npv10                  3                                                        
191              outer.rho_star.const_z.n40.npv10               10.0                                             Grenzzyklus
192                       outer.plateau.n40.npv20                  3                                                        
193              outer.rho_star.const_z.n40.npv20               10.0                                             Grenzzyklus
194                      outer.plateau.n120.npv12                  3                                                        
195                 outer.k_at_rho_max.n120.npv12               43.0                                                        
196                      outer.plateau.n120.npv30                  3                                                        
197                 outer.k_at_rho_max.n120.npv30               36.0                                                        
198                      outer.plateau.n120.npv60                  3                                                        
199                 outer.k_at_rho_max.n120.npv60               35.0                                                        
200                      outer.plateau.n350.npv35                  4                                                        
201             outer.rho_star.const_z.n350.npv35               3.16                                               Divergenz
202                      outer.plateau.n350.npv88                  3                                                        
203             outer.rho_star.const_z.n350.npv88               4.64                                               Divergenz
204                     outer.plateau.n350.npv175                  3                                                        
205            outer.rho_star.const_z.n350.npv175               4.64                                               Divergenz
206                          fail.count.Divergenz                 10                                                        
207                          fail.eps_V.Divergenz  3.56e-01-8.92e-01  p.u.                                                  
208                          fail.q_max.Divergenz       617-4.67e+04  p.u.                                                  
209                        fail.eps_lin.Divergenz        0.999-1.014                                                        
210                        fail.count.Grenzzyklus                  3                                                        
211                        fail.eps_V.Grenzzyklus  9.64e-07-2.58e-05  p.u.                                                  
212                        fail.q_max.Grenzzyklus          6.22-17.1  p.u.                                                  
213                      fail.eps_lin.Grenzzyklus        0.916-0.968                                                        
214         pred.spearman_outer.sens_error_median              0.899                                                        
215         pred.spearman_inner.sens_error_median              0.892                                                        
216            pred.spearman_outer.sens_error_max              0.898                                                        
217            pred.spearman_inner.sens_error_max              0.885                                                        
218                        pred.spearman_outer.rx              0.537                                                        
219                        pred.spearman_inner.rx              0.511                                                        
220               pred.spearman_outer.q_max_final               0.35                                                        
221               pred.spearman_inner.q_max_final              0.249                                                        
222                     pred.spearman_outer.v_min             -0.329                                                        
223                     pred.spearman_inner.v_min             -0.593                                                        
224                pred.spearman_outer.rho_jacobi              -0.24                                                        
225                pred.spearman_inner.rho_jacobi             -0.089                                                        
226                  pred.spearman_outer.cond_xpp             -0.239                                                        
227                  pred.spearman_inner.cond_xpp             -0.087                                                        
228                      pred.spearman_outer.n_pv              -0.19                                                        
229                      pred.spearman_inner.n_pv              0.038                                                        
230                   pred.spearman_outer.eta_pub              0.155                                                        
231                   pred.spearman_inner.eta_pub              0.447                                                        
232                     pred.spearman_outer.nodes             -0.068                                                        
233                     pred.spearman_inner.nodes              0.222                                                        
234                                   pred.n_runs                217                        konvergente PV-Laeufe, alle Modi
235                               nr.faelle.beide                191                                                        
236                             nr.vmin_med.beide              0.978  p.u.                                                  
237                             nr.eta2_med.beide              0.035                                                        
238                              nr.rho_min.beide                0.1                                                        
239                              nr.faelle.keines                 41                                                        
240                            nr.vmin_med.keines              0.884  p.u.                                                  
241                            nr.eta2_med.keines              0.142                                                        
242                             nr.rho_min.keines               2.15                                                        
243                              nr.faelle.nur NR                  2                                                        
244                            nr.vmin_med.nur NR              0.989  p.u.                                                  
245                            nr.eta2_med.nur NR              0.012                                                        
246                             nr.rho_min.nur NR               10.0                                                        
247                             nr.faelle.nur TPF                  0                                                        
248                                     nr.n_runs                234                        gekoppelte PV-Laeufe beider Modi
249                          nr.skipped_in_keines                  0                             nicht konstruierbare Faelle
```

## tab_rx_inner

```
   $n$     Modus  $\eta(0{,}1)$  $\eta(10)$  $k(0{,}1)$  $k(10)$
0   40  const\_z       0.013023    0.013304         7.0      7.0
1  120  const\_z       0.033869    0.035827         9.0      9.0
2  350  const\_z       0.109340    0.133063        13.0     14.0
3   40  const\_x       0.012921    0.178445         7.0     16.0
4  120  const\_x       0.033602         NaN         9.0      NaN
5  350  const\_x       0.108433         NaN        13.0      NaN
```

## tab_rx_outer_struct

```
   $n$   pv  $n_\mathrm{pv}$  $\mathrm{cond}$  $\rho_{\mathrm{J}}$  $\bar{x}_{kk}$
0   40  0.1                4         3.732051             0.767592        0.014919
1   40  0.5               20        80.308480             5.226260        0.012749
2  120  0.1               12        10.571898             1.184226        0.019440
3  120  0.5               60       303.561466            13.499865        0.017089
4  350  0.1               35        45.213818             3.577625        0.024335
5  350  0.5              175       975.535945            33.131058        0.021855
```

## tab_rx_outer

```
   $n$  $n_\mathrm{pv}$  $\le0{,}68$  $\rho=2{,}15$  $\rho=6{,}81$  $\rho^\ast$ Versagensart
0   40                4            3            4.0            7.0      10.0000  Grenzzyklus
1   40               10            3            4.0            7.0      10.0000  Grenzzyklus
2   40               20            3            4.0            7.0      10.0000  Grenzzyklus
3  120               12            3            3.0            7.0          NaN         {--}
4  120               30            3            3.0            7.0          NaN         {--}
5  120               60            3            4.0            7.0          NaN         {--}
6  350               35            4            9.0            NaN       3.1623    Divergenz
7  350               88            3            7.0            NaN       4.6416    Divergenz
8  350              175            3            6.0            NaN       4.6416    Divergenz
```

## tab_rx_outer_fail

```
    $n$  $n_\mathrm{pv}$   $\rho$  $\varepsilon_V$  $\max_k\lvert Q_k\rvert$  $k_{\mathrm{in}}/k_{\mathrm{out}}$  $\varepsilon_{\mathrm{lin}}$                    Verhalten
0    40                4  10.0000     2.583761e-05                  6.218514                            5.166667                      0.968098                  Grenzzyklus
1    40               10  10.0000     3.047794e-06                  8.721219                            5.166667                      0.935603                  Grenzzyklus
2    40               20  10.0000     9.635502e-07                 17.106725                            5.033333                      0.916410  Grenzzyklus\rlap{$^{\ast}$}
3   350               35   3.1623     5.308999e-01                616.805909                          114.566667                      1.013919                    Divergenz
4   350               35   4.6416     5.859992e-01               2874.451619                          190.750000                      1.003364                    Divergenz
5   350               35   6.8129     4.463735e-01               2479.195422                          193.833333                      1.001526                    Divergenz
6   350               35  10.0000     6.646402e-01               2515.930254                          196.816667                      1.002493                    Divergenz
7   350               88   4.6416     3.557839e-01               2894.000927                          187.700000                      1.006998                    Divergenz
8   350               88   6.8129     8.435412e-01               4382.982663                          193.766667                      1.002563                    Divergenz
9   350               88  10.0000     5.078707e-01              15554.663779                          194.433333                      1.000587                    Divergenz
10  350              175   4.6416     8.923366e-01              16759.285210                          193.833333                      0.999865                    Divergenz
11  350              175   6.8129     4.384217e-01              17069.594012                          196.816667                      1.006669                    Divergenz
12  350              175  10.0000     4.201996e-01              46660.757747                          196.816667                      0.999313                    Divergenz
```

## tab_rx_predictor

```
                       Gr\"o\ss e  $\rho_{\mathrm{S}}$ zu $k_{\mathrm{out}}$  $\rho_{\mathrm{S}}$ zu $k_{\mathrm{in}}$
0  Linearisierungsfehler (Median)                                   0.899296                                  0.892372
1    Linearisierungsfehler (Max.)                                   0.898370                                  0.884863
2                      $\rho=R/X$                                   0.537191                                  0.510669
3        $\max_k\lvert Q_k\rvert$                                   0.349585                                  0.248510
4                      $v_{\min}$                                  -0.329065                                 -0.593408
5             $\rho_{\mathrm{J}}$                                  -0.239851                                 -0.089488
6  $\mathrm{cond}(\vect{X}_{pp})$                                  -0.238683                                 -0.086646
7                 $n_\mathrm{pv}$                                  -0.190069                                  0.037871
8                          $\eta$                                   0.154968                                  0.446834
9                             $n$                                  -0.067772                                  0.222087
```

## tab_rx_nr

```
  konvergiert  F\"alle  $v_{\min}$ (Med.)  $v_{\min}^{\mathrm{NR}}$  $\eta_2$ (Med.)  $\rho_{\min}$  $\rho_{\max}$
0       beide      191           0.977944                  0.981029         0.035124         0.1000           10.0
1      keines       41           0.884203                       NaN         0.141564         2.1544           10.0
2      nur NR        2           0.988818                  0.993038         0.012161        10.0000           10.0
3     nur TPF        0                NaN                       NaN              NaN            NaN            NaN
```
