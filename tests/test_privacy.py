import numpy as np
import pandas as pd

from flowsight.privacy import suppress_small_groups


def test_small_groups_blanked_large_kept_and_input_untouched():
    t = pd.DataFrame(dict(zone=["a", "b"], visitors=[4, 5], visits=[4, 9], mean_dwell_s=[10.0, 12.0],
                          engaged_s=[30.0, 40.0], dead_zone=[True, False]))
    out = suppress_small_groups(t)
    assert np.isnan(out.loc[0, "mean_dwell_s"]) and np.isnan(out.loc[0, "engaged_s"])
    assert out.loc[1, "mean_dwell_s"] == 12.0 and out.loc[0, "visits"] == 4        # counts stay visible
    assert bool(out.loc[0, "suppressed"]) and not bool(out.loc[1, "suppressed"])
    assert not out.loc[0, "dead_zone"]                                            # too little data to call it
    assert t.loc[0, "mean_dwell_s"] == 10.0                                       # original not modified
