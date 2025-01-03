function weight = Polynomial_kernel(x)
    alpha = 1;  
    c = 0;     
    d = 2;      
    weight = (alpha * x + c) .^ d;
end