function CSTA = direct_CSTA( datarun , cone_params )

N_cones = numel( datarun.cones.types ) ;
N_GC    = numel( datarun.cell_ids    ) ;

% Unpacking GC_stas into: STA, norms of STAs and N_spikes
STA_norm = zeros(N_GC,1) ;
N_spikes = zeros(N_GC,1) ;
STA      = zeros(M0*M1*N_colors,N_GC) ;
for i=1:N_GC
    N_spikes(i) = length(GC_stas(i).spikes) ;
    STA(:,i)    = GC_stas(i).spatial(:) ;
    STA_norm(i) = norm(STA(:,i)) ;
end


% memoized(?) function returning gaussian mass in a box
gaus_in_box = gaus_in_a_box( cone_params.sigma ) ;


CSTA = zeros(N_cones*N_colors,N_GC) ;
fprintf('Calculating Cone_STA...\n')
for ii=1:N_cones
    i   = centers(ii,1) ;
    j   = centers(ii,2) ;
    BW  = make_filter(M0,M1,i,j,cone_params.support_radius,gaus_in_box) ;
    
    for c=1:N_colors
        filter  = kron(cone_params.colors(c,:),BW(:)') ;
        CSTA((c-1)*NROI + ii , :) = filter * STA ;
    end
end

end


function filter = make_filter(M0,M1,i,j,support,gaus_boxed)

filter = zeros(M0,M1) ;
ox  = max(1,floor(i-support)):min(M0,ceil(i+support)) ;
oy  = max(1,floor(j-support)):min(M1,ceil(j+support)) ;
x   = repmat(ox(:),numel(oy),1) ;
y   = reshape( repmat(oy,numel(ox),1) , [] , 1 ) ;
g   = gaus_boxed(i-x,j-y) ;
filter(x+(y-1)*M0) = g ;

end


function gf = gaus_in_a_box(sigma)

% memo    = sparse([],[],[],1000,1000, ceil( (SS*3*sigma)^2 ) ) ;
gf      = @gib ;

    function out = gib(dx,dy)
        
%         out = zeros(size(in));  % preallocate output
%         [tf,loc] = ismember(in,x);  % find which in's already computed in x
%         ft = ~tf;  % ones to be computed
%         out(ft) = F(in(ft));  % get output values for ones not already in
%         % place new values in storage
%         x = [x in(ft(:).')];
%         y = [y reshape(out(ft),1,[])];
%         out(tf) = y(loc(tf));  % fill in the rest of the output values

        dx  = dx(:) ;
        dy  = dy(:) ;
        l   =  ones(size(dx)) ;
        O   = zeros(size(dx)) ;
        
        out = mvncdf([dx dy] + [l l],[],sigma.*[1 1]) ...
            - mvncdf([dx dy] + [O l],[],sigma.*[1 1]) ...
            - mvncdf([dx dy] + [l O],[],sigma.*[1 1]) ...
            + mvncdf([dx dy]        ,[],sigma.*[1 1]) ;
    end

end