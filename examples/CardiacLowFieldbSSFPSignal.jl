using  KomaMRI, PlotlyJS
#Phantom
B0 = 0.55
fat_ppm = 3.4e-6
myocard = Phantom{Float64}(x=[0], T1=[600e-3], T2=[60e-3])
blood =   Phantom{Float64}(x=[0], T1=[1200e-3],T2=[410e-3])
fat =     Phantom{Float64}(x=[0], T1=[217e-3], T2=[95e-3], Δw=[-2π*γ*B0*fat_ppm])
obj = myocard+blood+fat
#Sequence parameters
TR = 5.81e-3
TE = TR / 2 #bSSFP condition
RR = 1 #1 [s]
Trf = 1e-4  #1 [ms]
B1 = 1 / (360*γ*Trf)
iNAV_lines = 14
im_segments = 20
iNAV_flip_angle = 3.2
im_flip_angle = 90
number_heart_beats = 1

function iNAV_bSSFP_cardiac(number_heart_beats, iNAV_lines, im_segments, iNAV_flip_angle, im_flip_angle)
    k = 0
    seq = Sequence()
    for hb = 0 : number_heart_beats - 1
        for i = 0 : iNAV_lines + im_segments - 1
            if iNAV_lines != 0
                m = (im_flip_angle - iNAV_flip_angle) / iNAV_lines
                α = min( m * i + iNAV_flip_angle, im_flip_angle ) * exp(1im * π * k)
            else
                α = im_flip_angle * exp(1im * π * k)
            end
            seq += RF(α * B1, Trf)
            seq += Delay(TR)
            k += 1
        end
        seq += Delay(RR - TR * (iNAV_lines + im_segments))
    end
    return seq
end

αs = max.(1:10:180, iNAV_flip_angle)
M = zeros(ComplexF64, length(αs), 3)
for (j, im_flip_angle) = enumerate(αs)
    #Sequence
    seq = iNAV_bSSFP_cardiac(number_heart_beats, iNAV_lines, im_segments, iNAV_flip_angle, im_flip_angle)
    #Simulation
    sys = Scanner()
    simParams = Dict{String,Any}("Nblocks"=>1, "return_type"=>"state")
    aux = simulate(obj, seq, sys; simParams)
    M[j,1] = aux.xy[1]
    M[j,2] = aux.xy[2]
    M[j,3] = aux.xy[3]
end
myo_plt = scatter(x=αs, y=abs.(M[:,1]), name="Myocardium")
blo_plt = scatter(x=αs, y=abs.(M[:,2]), name="Blood")
fat_plt = scatter(x=αs, y=abs.(M[:,3]), name="Fat")
plot([myo_plt, blo_plt, fat_plt], Layout(
    title=attr(
        text= "Optimal Flip Angle for bSSFP at 0.55T",
    ),
    xaxis_title="Flip Angle [deg]",
    yaxis_title="Signal Intensity [a.u.]"
))