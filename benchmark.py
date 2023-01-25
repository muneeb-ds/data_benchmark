from operations import *

_, stats_pd = pd_read_csv("filtered_agents.csv")
_, stats_md = md_read_csv("filtered_agents.csv")
_, stats_pl = pl_read_csv("filtered_agents.csv")

print(stats_pd)
print(stats_md)
print(stats_pl)