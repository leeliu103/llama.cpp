#pragma once

struct gptoss_decode_aql;
struct gptoss_decode_layer_params;

gptoss_decode_aql * gptoss_decode_aql_create(int multiprocessor_count);

void gptoss_decode_aql_destroy(gptoss_decode_aql * aql);

bool gptoss_decode_aql_launch(
    gptoss_decode_aql * aql,
    const gptoss_decode_layer_params * params);
