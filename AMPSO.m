function record = AMPSO(RUN_TIMES, SIZE, DIM)

    global vel_percent;
    global Ner;
    global alpha1;
    global alpha2;
    global alpha3;
    global beta1;

    
    vel_percent = [0.01, 0.01, 0.01];
    Ner = 10; % the number of particles in every sub-swarm of exploration swarm
    alpha1 = 0.02; % control N1      
    alpha2 = 0.2; % control N2
    alpha3 = 0.25; % control Ns
    beta1 = 0.001;

    FUNC_LIST = 1:15;
    record = zeros(length(FUNC_LIST), RUN_TIMES);

    for func_num = 1:length(FUNC_LIST)
        for run = 1:RUN_TIMES
            func = FUNC_LIST(func_num);
            disp(['Func:', num2str(func), '    Run: ', num2str(run)]);
            record(func_num, run) = main(func, SIZE, DIM);
        end
    end
end


function best_fit = main(FUNC, SIZE, DIM)
    global alpha1;
    global alpha2;
    fhd = @get_value;
    
    LB = -100; UB = 100;
    fes_max = DIM * 10000;
    Ntotal = fes_max / SIZE;
    
    swarm = initial(fhd, FUNC, SIZE, DIM, LB, UB);
    search_swarms = swarm{1}; exploit_swarm = swarm{2}; convergence_swarm = swarm{3};
    
    Ni = 0; epoch = 0; 
    % search coefficient
    N1 = alpha1 * Ntotal;   N2 = alpha2 * Ntotal;
    swarm2_fit = Inf;   swarm2_pos = zeros(1, DIM);
    while Ni < ceil(Ntotal / 3)
        search_swarms = swarm1(search_swarms, N1);
        Ni = Ni + N1 + 1;
        swarm1_fit = Inf;   swarm1_pos = zeros(1, DIM);
        for index = 1:length(search_swarms)
            if search_swarms{index}.solution.best_fit < swarm1_fit
                swarm1_fit = search_swarms{index}.solution.best_fit;
                swarm1_pos = search_swarms{index}.solution.best_pos;
            end
        end

        [exploit_swarm, step_exploit] = swarm2(exploit_swarm, swarm1_pos, N1, N2);
        Ni = Ni + step_exploit + 1;
        if exploit_swarm.solution.best_fit < swarm2_fit
            swarm2_fit = exploit_swarm.solution.best_fit;
            swarm2_pos = exploit_swarm.solution.best_pos;
        end

        epoch = epoch + 1;
        % disp(['epoch:', num2str(epoch)]);
    end
    
    remain_Ni = Ntotal - Ni;
    convergence_swarm = swarm3(convergence_swarm, swarm2_pos, remain_Ni);
    % disp(['the best fit: ', num2str(convergence_swarm.solution.best_fit)]);
    % convergence_swarm.solution.best_pos
    best_fit = convergence_swarm.solution.best_fit;
end


function swarm = initial(fhd, FUNC, SIZE, DIM, LB, UB)
    global vel_percent;
    global Ner;
    
    swarm = cell(1,3);
    N = ceil(SIZE / Ner);
    swarm{1}=cell(1,N);
    for index = 1:N 
        swarm{1}{index}.setting =  struct('fhd', fhd, 'func', FUNC, 'size', Ner, 'dim', DIM, ...
            'lb', LB, 'ub', UB, 'v_max', (UB-LB)*vel_percent(1), ...
            'w', 2.0, 'c1', 1.49445, 'c2', 1.49445, 'sigma', 0.0);
    end
    swarm{2}.setting = struct('fhd', fhd, 'func', FUNC, 'size', SIZE, 'dim', DIM,...
        'lb', LB, 'ub', UB, 'v_max', (UB-LB)*vel_percent(2), ...
        'w', 2.0, 'c1', 1.49445, 'c2', 1.49445, 'sigma', 0.0);
    swarm{3}.setting = struct('fhd', fhd, 'func', FUNC, 'size', SIZE, 'dim', DIM,...
        'lb', LB, 'ub', UB, 'v_max', (UB-LB)*vel_percent(3), ...
        'w', 2.0, 'c1', 1.49445, 'c2', 1.49445, 'sigma', 0.0);
    
    
    for index = 1:N 
        swarm{1}{index}.particles.pos = rand(swarm{1}{index}.setting.size, swarm{1}{index}.setting.dim) ...
            * (swarm{1}{index}.setting.ub - swarm{1}{index}.setting.lb) + swarm{1}{index}.setting.lb;
        swarm{1}{index}.particles.vel = (rand(swarm{1}{index}.setting.size, swarm{1}{index}.setting.dim)...
            * 2 - 1) * swarm{1}{index}.setting.v_max;
        swarm{1}{index}.particles.fit = feval(swarm{1}{index}.setting.fhd, swarm{1}{index}.particles.pos, swarm{1}{index}.setting.func);
        swarm{1}{index}.particles.best_pos = swarm{1}{index}.particles.pos;
        swarm{1}{index}.particles.best_fit = swarm{1}{index}.particles.fit;
    
        [~, min_index] = min(swarm{1}{index}.particles.best_fit);
        swarm{1}{index}.solution.best_pos = swarm{1}{index}.particles.best_pos(min_index, :);
        swarm{1}{index}.solution.best_fit = swarm{1}{index}.particles.best_fit(min_index);
        swarm{1}{index}.solution.best_index = min_index;
    end
    
    
    for index = 2:3            
        swarm{index}.particles.pos = rand(swarm{index}.setting.size, swarm{index}.setting.dim) ...
            * (swarm{index}.setting.ub - swarm{index}.setting.lb) + swarm{index}.setting.lb;
        swarm{index}.particles.vel = (rand(swarm{index}.setting.size, swarm{index}.setting.dim)...
            * 2 - 1) * swarm{index}.setting.v_max;
        swarm{index}.particles.fit = feval(swarm{index}.setting.fhd, swarm{index}.particles.pos, swarm{index}.setting.func);
        swarm{index}.particles.best_pos = swarm{index}.particles.pos;
        swarm{index}.particles.best_fit = swarm{index}.particles.fit;
        
        [~, min_index] = min(swarm{index}.particles.best_fit);
        swarm{index}.solution.best_pos = swarm{index}.particles.best_pos(min_index, :);
        swarm{index}.solution.best_fit = swarm{index}.particles.best_fit(min_index);
        swarm{index}.solution.best_index = min_index;
    end
end


function swarms_new = swarm1(swarms, N1)   
    % reset search_swarm
    N = length(swarms);
    for index = 1:N
        swarm = swarms{index};
        swarm.setting.w = 1.0; 
        swarm.setting.c1 = 2.0; 
        swarm.setting.c2 = 2.0; 
        swarm.setting.sigma = 0.1;
    
        swarm.particles.pos = rand(swarm.setting.size, swarm.setting.dim) ...
            * (swarm.setting.ub - swarm.setting.lb) + swarm.setting.lb;
        swarm.particles.vel = (rand(swarm.setting.size, swarm.setting.dim) * 2 - 1) * swarm.setting.v_max;
        swarm.particles.fit = feval(swarm.setting.fhd, swarm.particles.pos, swarm.setting.func);
        swarm.particles.best_pos = swarm.particles.pos;
        swarm.particles.best_fit = swarm.particles.fit;
    
        [~, min_index] = min(swarm.particles.best_fit);
        swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
        swarm.solution.best_fit = swarm.particles.best_fit(min_index);
        swarm.solution.best_index = min_index;
        swarms{index} = swarm;
    end
    
    for iter = 1:N1
        % udate coefficients
        % Et = [0, 1]
        w_range = [0.65, 0.8]; 
        c1 = 1.49445;
        c2 = 1.49445;
        sigma_range = [0.1, 0.1];
        for index_out = 1:N 
            swarm = swarms{index_out};
            swarm = update_coe(swarm, w_range, sigma_range);
            swarm.setting.c1 = c1;
            swarm.setting.c2 = c2;
    
            % update position
            temp_vel = (swarm.setting.w * swarm.particles.vel) ...
                +(swarm.setting.c1*rand(swarm.setting.size,swarm.setting.dim).*(swarm.particles.best_pos-swarm.particles.pos)) ...
                +(swarm.setting.c2*rand(swarm.setting.size,swarm.setting.dim).*(ones(swarm.setting.size,1)*swarm.solution.best_pos-swarm.particles.pos));
            temp_vel = max(-swarm.setting.v_max, min(swarm.setting.v_max, temp_vel));
            swarm.particles.vel = temp_vel;
            temp_pos = swarm.particles.pos + swarm.particles.vel;
            temp_pos = max(swarm.setting.lb, min(swarm.setting.ub, temp_pos));
            swarm.particles.pos = temp_pos;
            swarm.particles.fit = feval(swarm.setting.fhd, swarm.particles.pos, swarm.setting.func);
            for index = 1:swarm.setting.size
                if swarm.particles.fit(index) < swarm.particles.best_fit(index)
                    swarm.particles.best_pos(index, :) = swarm.particles.pos(index, :);
                    swarm.particles.best_fit(index) = swarm.particles.fit(index);
                end
            end
    
            [~, min_index] = min(swarm.particles.best_fit);   % note: the best_fit is used here
            swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
            swarm.solution.best_fit = swarm.particles.best_fit(min_index);
            swarm.solution.best_index = min_index;
            swarms{index_out} = swarm;
        end
        
    end
    swarms_new = swarms;
end


function [swarm_new, step] = swarm2(swarm, original_pos, iter_exploit_min, iter_exploit_max)    
    global alpha3
    
    %% generation of exploit swarm
    sigma = 0.1;
    position = zeros(swarm.setting.size, swarm.setting.dim);
    position(1, :) = original_pos;
    for index = 2:swarm.setting.size
        position(index, :) = disturbance(original_pos, swarm, sigma, 'r_dir');
    end
    position = max(swarm.setting.lb, min(swarm.setting.ub, position));
    swarm.particles.pos = position;
    swarm.particles.vel = (rand(swarm.setting.size, swarm.setting.dim)...
        * 2 -1) * swarm.setting.v_max;
    swarm.particles.fit = feval(swarm.setting.fhd, swarm.particles.pos, swarm.setting.func);
    swarm.particles.best_pos = swarm.particles.pos;
    swarm.particles.best_fit = swarm.particles.fit;
    
    [~, min_index] = min(swarm.particles.best_fit);   % note: the best_fit is used here
    swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
    swarm.solution.best_fit = swarm.particles.best_fit(min_index);
    swarm.solution.best_index = min_index;
    
    %% iteration
    step = 1;
    record = zeros(1, 1000); record(1) = swarm.solution.best_fit;
    while true
        % update coefficients
        % Et = [0, 1]
        w_range = [0.5, 0.65]; 
        swarm.setting.c1 = 1.49445;
        swarm.setting.c2 = 1.49445;
        sigma_range = [0.1, 0.2];
        swarm = update_coe(swarm, w_range, sigma_range);        
        % rebuilding of particle
        num_p1 = ceil(alpha3 * swarm.setting.size);  % the number of rebuilding particles
        num_p2 = swarm.setting.size - num_p1;
        [~, sort_index] = sort(swarm.particles.best_fit);
        worst_index = sort_index(num_p2+1:swarm.setting.size);
        position = zeros(num_p1, swarm.setting.dim);
        for index = 1:num_p1
            position(index, :) = disturbance(swarm.solution.best_pos, swarm, swarm.setting.sigma, '1_dim');
        end
        position = max(swarm.setting.lb, min(swarm.setting.ub, position));
        fitness = feval(swarm.setting.fhd, position, swarm.setting.func);
        swarm.particles.pos(worst_index, :) = position;
        swarm.particles.fit(worst_index) = fitness;
        for index = 1:swarm.setting.size
            if swarm.particles.fit(index) < swarm.particles.best_fit(index)
                swarm.particles.best_pos(index, :) = swarm.particles.pos(index, :);
                swarm.particles.best_fit(index) = swarm.particles.fit(index);
            end
        end
        
        [~, min_index] = min(swarm.particles.best_fit);   % note: the best_fit is used here
        swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
        swarm.solution.best_fit = swarm.particles.best_fit(min_index);
        swarm.solution.best_index = min_index;
    
        % update swarm
        select_index = randperm(swarm.setting.size);
        select_index = select_index(1:num_p2);
        temp_vel = swarm.particles.vel(select_index, :);
        temp_pos = swarm.particles.pos(select_index, :);
        temp_best_pos = swarm.particles.best_pos(select_index, :);
    
        temp_vel = swarm.setting.w * temp_vel ...
            + swarm.setting.c1 * rand(num_p2, swarm.setting.dim) .* (temp_best_pos - temp_pos) ...
            + swarm.setting.c2 * rand(num_p2, swarm.setting.dim) .* (ones(num_p2, 1) * swarm.solution.best_pos - temp_pos);
        temp_vel = max(-swarm.setting.v_max, min(swarm.setting.v_max, temp_vel));
        temp_pos = temp_pos + temp_vel;
        temp_pos = max(swarm.setting.lb, min(swarm.setting.ub, temp_pos));
        temp_fit = feval(swarm.setting.fhd, temp_pos, swarm.setting.func);
    
        swarm.particles.vel(select_index, :) = temp_vel;
        swarm.particles.pos(select_index, :) = temp_pos;
        swarm.particles.fit(select_index) = temp_fit;
    
        for index = 1:swarm.setting.size
            if swarm.particles.fit(index) < swarm.particles.best_fit(index)
                swarm.particles.best_pos(index, :) = swarm.particles.pos(index, :);
                swarm.particles.best_fit(index) = swarm.particles.fit(index);
            end
        end
    
        [~, min_index] = min(swarm.particles.best_fit);   % note: the best_fit is used here
        swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
        swarm.solution.best_fit = swarm.particles.best_fit(min_index);
        swarm.solution.best_index = min_index;
    
        step = step + 1;
        record(step) = swarm.solution.best_fit;
        
        terminal_flag = terminal_conditions(record, step, iter_exploit_min, iter_exploit_max);
        if terminal_flag
            break
        end        
    end
    swarm_new = swarm;
end


function swarm_new = swarm3(swarm, original_pos, remain_Ni)  
    global beta1;
    
    %% generation of exploit swarm
    sigma = 0.1;
    position = zeros(swarm.setting.size, swarm.setting.dim);
    position(1, :) = original_pos;
    for index = 2:swarm.setting.size
        position(index, :) = disturbance(original_pos, swarm, sigma, 'r_dir');
    end
    position = max(swarm.setting.lb, min(swarm.setting.ub, position));
    swarm.particles.pos = position;
    swarm.particles.vel = (rand(swarm.setting.size, swarm.setting.dim)...
        * 2 -1) * swarm.setting.v_max;
    swarm.particles.fit = feval(swarm.setting.fhd, swarm.particles.pos, swarm.setting.func);
    swarm.particles.best_pos = swarm.particles.pos;
    swarm.particles.best_fit = swarm.particles.fit;
    
    [~, min_index] = min(swarm.particles.best_fit);   % note: the best_fit is used here
    swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
    swarm.solution.best_fit = swarm.particles.best_fit(min_index);
    swarm.solution.best_index = min_index;
    disp(['best generated fit:  ', num2str(swarm.solution.best_fit)])
    
    %% iteration
    step = 1; t0 = 0; t1 = 0; not_upgrade = 0;
    record = zeros(1, ceil(remain_Ni+1));
    record(1) = swarm.solution.best_fit;
    while step < remain_Ni
        % update coefficient
        % Et = [0, 1]
        w_range = [0.5, 0.8]; 
        swarm.setting.c1 = 1.49445;
        swarm.setting.c2 = 1.49445;
        sigma_range = [0.1, 0.2];
        swarm = update_coe(swarm, w_range, sigma_range);

        pc = 1.0 / (1 + exp(25 - not_upgrade));

        if rand < pc
            t0 = t0 + 1;  
            not_upgrade = 0;
            % rebuild swarm
            position = zeros(swarm.setting.size, swarm.setting.dim);
            for index = 1:swarm.setting.size
                position(index, :) = disturbance(swarm.solution.best_pos, swarm, swarm.setting.sigma, '1_dim');
            end
            position = max(swarm.setting.lb, min(swarm.setting.ub, position));
            swarm.particles.pos = position;
            swarm.particles.fit = feval(swarm.setting.fhd, position, swarm.setting.func);
            for index = 1:swarm.setting.size
                if swarm.particles.fit(index) < swarm.particles.best_fit(index)
                    swarm.particles.best_pos(index, :) = swarm.particles.pos(index, :);
                    swarm.particles.best_fit(index) = swarm.particles.fit(index);
                end
            end
    
            [~, min_index] = min(swarm.particles.best_fit);   % note: the best_fit is used here
            swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
            swarm.solution.best_fit = swarm.particles.best_fit(min_index);
            swarm.solution.best_index = min_index;
        
        else
            t1 = t1 + 1;

            temp_vel = (swarm.setting.w * swarm.particles.vel) ...
                +(swarm.setting.c1*rand(swarm.setting.size,swarm.setting.dim).*(swarm.particles.best_pos-swarm.particles.pos)) ...
                +(swarm.setting.c2*rand(swarm.setting.size,swarm.setting.dim).*(ones(swarm.setting.size,1)*swarm.solution.best_pos-swarm.particles.pos));
            temp_vel = max(-swarm.setting.v_max, min(swarm.setting.v_max, temp_vel));
            swarm.particles.vel = temp_vel;
            temp_pos = swarm.particles.pos + swarm.particles.vel;
            temp_pos = max(swarm.setting.lb, min(swarm.setting.ub, temp_pos));
            swarm.particles.pos = temp_pos;
            swarm.particles.fit = feval(swarm.setting.fhd, swarm.particles.pos, swarm.setting.func);
            for index = 1:swarm.setting.size
                if swarm.particles.fit(index) < swarm.particles.best_fit(index)
                    swarm.particles.best_pos(index, :) = swarm.particles.pos(index, :);
                    swarm.particles.best_fit(index) = swarm.particles.fit(index);
                end
            end
    
            [~, min_index] = min(swarm.particles.best_fit);
            swarm.solution.best_pos = swarm.particles.best_pos(min_index, :);
            swarm.solution.best_fit = swarm.particles.best_fit(min_index);
            swarm.solution.best_index = min_index;
        end
    
    
        step = step + 1;
        record(step) = swarm.solution.best_fit;

        Er = evolution_rate(record, step, remain_Ni);
        if Er < beta1
            not_upgrade = not_upgrade + 1;
        else
            not_upgrade = 0;
        end
        swarm_new = swarm;
    end     
end

    
function flag = terminal_conditions(record, step, iter_exploit_min, iter_exploit_max)
    global beta1;

    flag = 0;
    if step > iter_exploit_max
        flag = 1;
    end
    if evolution_rate(record, step, iter_exploit_max) < beta1
        flag = 1;
    end
    if step < iter_exploit_min
        flag = 0;
    end
end


function Er = evolution_rate(record, t, N)
    if N > 1000
        K = 50;
    else
        K = 10;
    end

    if t > K
        Er = (record(t-K) - record(t)) /  (K * record(t-1));
    elseif t == 1
        Er = 1;
    else
        Er = (record(1) - record(t)) / (t * record(t-1));
    end
end

function fitness = get_value(position, func_num)
    handle = str2func('cec15_func');
    fitness = feval(handle, position', func_num);
    f = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500];
    fitness = fitness - f(func_num);
end

function pos_new = disturbance(pos, swarm, sigma, choose)

    if strcmp(choose, 'r_dir')
        dr = normrnd(0,sigma,[1,swarm.setting.dim]);
        pos_new = pos + (swarm.setting.ub - swarm.setting.lb) * dr;
        
    elseif strcmp(choose, '1_dim')
        dim = randi(swarm.setting.dim);
        pos(dim) = pos(dim) + normrnd(0.0, sigma, [1,1]) * (swarm.setting.ub - swarm.setting.lb);
        pos_new = pos;
        
    elseif strcmp(choose, 'rand_dim')
        dim_num = randi(ceil(swarm.setting.dim / 5));
        for index = 1:dim_num
            dim = randi(swarm.setting.dim);
            pos(dim) = pos(dim) + normrnd(0.0, sigma, [1,1]) * (swarm.setting.ub - swarm.setting.lb);
        end
        pos_new = pos;
        
    else
        disp('input error')
    end
end


function swarm_new = update_coe(swarm, w_range, sigma_range)
    
    Et = diversity(swarm);
    a = 1.0;
    swarm.setting.w = transform_coe_update(w_range, a, Et);
    swarm.setting.sigma = (sigma_range(2)-sigma_range(1)) * Et + sigma_range(1);
    swarm_new = swarm;
end
    
function coe = transform_coe_update(x_range, a, Et)
    b = 1/x_range(1) - a;
    c = log((1.0/x_range(2) - a) / b);
    coe = 1 / (a + b * exp(c * Et));
end  
    
function diversity = diversity(swarm)
    N = swarm.setting.size;
    E_pos = 0;
    for index_1 = 1:swarm.setting.dim
        p = zeros(1, N);
        pos_tri = linspace(swarm.setting.lb, swarm.setting.ub, N);
        for index_2 = 1:swarm.setting.size
            handle = find(swarm.particles.pos(index_2, index_1) <= pos_tri, 1);
            p(handle) = p(handle) + 1;
        end
        p = p / swarm.setting.size;
        p(p==0) = [];
        entropy_temp = 0;
        for index_3 = 1:length(p)
            entropy_temp = entropy_temp - p(index_3) * log(p(index_3));
        end
        entropy_temp = entropy_temp / log(N);
        E_pos = entropy_temp / swarm.setting.dim;
    end

    fitness = swarm.particles.fit;
    fit_tri = linspace(min(fitness), max(fitness), N);
    p = zeros(1, N);
    for index = 1:swarm.setting.size
        handle = find(fitness(index) <= fit_tri, 1);
        p(handle) = p(handle) + 1;
    end
    p = p / sum(p);
    p(p==0) = [];
    E_fit = 0;
    for index = 1:length(p)
        E_fit = E_fit - p(index) * log(p(index));
    end
    E_fit = E_fit / log(N);

    diversity = (E_pos + E_fit) / 2;
end