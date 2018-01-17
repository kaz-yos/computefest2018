data {
    int <lower=1> N;
    int <lower=1> M;
    matrix[M, N] X;
    int y[N];
}

parameters {
    vector[M] beta;
    real alpha;
}

model {
    beta ~ normal(0,5);
    alpha ~ normal(0, 5);
    y ~ bernoulli_logit (X' * beta + alpha);
}

generated quantities {
    real p_hat_ppc = 0;

    for (n in 1:N) {
        int y_ppc = bernoulli_logit_rng(X[1:M, n]' * beta + alpha);
        p_hat_ppc += y_ppc;
    }

    p_hat_ppc = p_hat_ppc / N;
}
