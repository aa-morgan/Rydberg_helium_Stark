clear all

data_80_83 = importdata('.\StarkMapData_n_80-83.csv');
field_80_83 = importdata('.\StarkMapField_n_80-83.csv');
states_80_83 = importdata('.\StarkMapStates_n_80-83.csv');
n_vals = states_80_83(1,:);
l_vals = states_80_83(2,:);
m_vals = states_80_83(3,:);

alpha = 0.0072973525664;
m_e = 9.10938356e-31;
c = 299792458.0;
h = 6.62607004e-34;
En_h = alpha^2.0 * m_e * c^2.0;
E_h = 4.35974465054e-18;
scl = c*10^-9* En_h /(h * c);

PLOT_SELECTED_M_VALS = true;
if PLOT_SELECTED_M_VALS == true
    colours = ['b', 'r', 'c', 'm', 'k'];
    m_vals_to_plot = [0,1];
    i=0;
    for m_val = m_vals_to_plot
        idx = find(m_vals==m_val);
        plot(field_80_83, data_80_83(idx,:)*scl, 'color', colours( mod( i, length(colours))+1 ) );
        hold on
        i=i+1;
    end
else
    plot(field_80_83, data_80_83*scl, 'color', 'b' );
end

xlabel('Field (V/cm)');
ylabel('Energy/h (GHz)');
xlim([0.0 0.525]);