function stat = GetShapeStatistics(mag)

stat = zeros(1, 4);
mom = rawMoment(mag, 4);

% Centroid
stat(1) = mom(1);

% Spread
stat(2) = sqrt(mom(2) - mom(1)^2);

% Skewness
stat(3) = (2*mom(1)^3 - 3*mom(1)*mom(2) + mom(3))/(stat(2)^3);

% Kurtosis
stat(4) = (-3*mom(1)^4 + 6*mom(1)*mom(2) - 4*mom(1)*mom(3) + mom(4))/(stat(2)^4) - 3;

end

function mom = rawMoment(mag, n)

num = zeros(1, 4);

for k = 1:length(mag)
    for m = 1:4
        num(m) = k^m * mag(k);
    end
end

mom = num/sum(mag);

end