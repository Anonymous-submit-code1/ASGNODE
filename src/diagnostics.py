"""Measurements in the actual reversible random-walk geometry."""
import torch


@torch.no_grad()
def field_diagnostics(model, data, position_encoding, terminal_time):
    model.eval()
    model.reset_nfe()
    times = data.x.new_tensor([0., terminal_time/2., terminal_time])
    states = model.trajectory(data.x, position_encoding, times)
    function = model.odeblock.odefunc
    degrees = data.degrees[:, None]
    mass = degrees.sum()
    initial_centroid = (degrees*states[0]).sum(0)/mass
    initial_energy = .5*(degrees*(states[0]-initial_centroid).square()).sum()
    result = []
    for t, state in zip(times, states):
        comp = function.velocity_components(state)
        centroid = (degrees*state).sum(0)/mass
        centered = state-centroid
        attractive_loss = -(degrees*centered*comp['attractive']).sum()
        separation_gain = -(degrees*centered*comp['separation_constraint']).sum()
        norm_a = torch.linalg.vector_norm(comp['attractive'])
        norm_s = torch.linalg.vector_norm(comp['separation_constraint'])
        parameters = function.get_stability_parameters(state)
        result.append({
            'time': float(t),
            'Q': float(norm_s/norm_a) if norm_a > 1e-12 else None,
            'attractive_norm': float(norm_a), 'separation_norm': float(norm_s),
            'energy_offset_ratio': float(separation_gain/attractive_loss) if attractive_loss > 1e-12 else None,
            'energy_ratio_bound': float(parameters.budget_ratio) if function.field_mode == 'certified' else None,
            'normalized_disagreement_energy': float(.5*(degrees*centered.square()).sum()/initial_energy) if initial_energy > 1e-12 else None,
            'weighted_centroid_drift': float(torch.linalg.vector_norm(centroid-initial_centroid)),
            'min_channel_gain': float((parameters.kappa+torch.nn.functional.softplus(function.alpha_train)).min()),
            'source_norm': float(torch.linalg.vector_norm(comp['source'])),
        })
    model.reset_nfe()
    return result
