data {
    int <lower=1> N;
    int <lower=1> M;
    matrix[M,N] X;
    int y[N];
}

parameters {
    vector[M] beta;
    real alpha;
    /* http://rstudio-pubs-static.s3.amazonaws.com/34099_2e35c3966ef548c2918d5b6c2146bfd1.html */
    real<lower=0> phi; // neg. binomial dispersion parameter
}

model {
    /* Priors */
    phi ~ cauchy(0, 3);
    beta ~ normal(0, 10);
    alpha ~ normal(0, 10);

    /* No need for link function here */
    y ~ neg_binomial_2_log(X' * beta + alpha, phi);
}

generated quantities {
    int y_ppc[N];
    for (n in 1:N)
        y_ppc[n] = neg_binomial_2_log_rng(X[1:M, n]' * beta + alpha, phi);
}
