data {
    int <lower=1> N;
    int <lower=1> M;
    matrix[M, N] X;
    int y[N];
    int <lower=1> N_hand;
    int <lower=1, upper=2> hand[N];
}

parameters {
    vector[M] beta;
    vector[N_hand] alpha;
}

model {
    beta ~ normal(0, 5);
    alpha ~ normal(0, 5);
    y ~ bernoulli_logit(X' * beta + alpha[hand]);
}

generated quantities {
    real p_hat_ppc = 0;
    real p_hat_left_ppc = 0;
    real p_hat_right_ppc = 0;

    /* This {} is temporary, and is not saved. */
    {

        int n_left = 0;
        int n_right = 0;

        for (n in 1:N) {
            int y_ppc = bernoulli_logit_rng(X[1:M, n]' * beta + alpha[hand[n]]);

            p_hat_ppc += y_ppc;
            if (hand[n] == 1) {
                p_hat_left_ppc += y_ppc;
                n_left += 1;
            } else {
                p_hat_right_ppc += y_ppc;
                n_right += 1;
            }
        }

        p_hat_ppc = p_hat_ppc / (n_left + n_right);
        p_hat_left_ppc = p_hat_left_ppc / n_left;
        p_hat_right_ppc = p_hat_right_ppc / n_right;
    }
}
