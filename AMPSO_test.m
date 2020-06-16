clc, clear, close all
rng('shuffle');

RUN_TIMES = 3;
SIZE = 40;
DIM = 10;

result = AMPSO(RUN_TIMES, SIZE, DIM);

savefile = ['AMPSO', '_SIZE', num2str(SIZE), '_DIM', num2str(DIM), '.mat'];
save(savefile, 'result');