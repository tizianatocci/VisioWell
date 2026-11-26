nextflow.enable.dsl = 2

// ==========================================================
// WORKFLOW DEFINITION
// ==========================================================
workflow {

    // --- Parameter Validation ---
    if (!params.exp_id)  { error "ERROR: --exp_id is required" }
    if (!params.sub_id)  { error "ERROR: --sub_id is required" }

    // Assigning parameters to local variables for clarity
    def czi_folder = params.czi_folder
    def exp_id = params.exp_id
    def sub_id = params.sub_id
    def rotation_angle = params.rotation_angle

    // Helper variables to store directory paths safely across steps
    def path_to_step0 = null
    def path_to_step1 = null

    // STEP 0: Find Wells and Pre-process
    if (params.run_steps == "step0" || params.run_steps == "all") {
        def step0_output = STEP0_PROCESS_CZI(czi_folder, exp_id, sub_id, rotation_angle)
        path_to_step0 = step0_output.folder
    }

    // STEP 1: Segmentation and curve correction
    if (params.run_steps == "step1" || params.run_steps == "all") {
        // Fallback: if Step 0 didn't run in this session, use the published folder
        if (!path_to_step0) {
            path_to_step0 = file("${params.outdir}/${exp_id}/${sub_id}/step0", type: 'dir')
        }

        def step1_output = STEP1_DETECTION(
            path_to_step0,
            exp_id,
            sub_id,
            params.which_network,
            params.name_experiment,
            params.training_to_load,
            params.flag_plot,
            params.flag_gt,
            params.positions ?: "",
            params.name_series_list ?: "",
            params.name_well_list ?: ""
        )
        path_to_step1 = step1_output.detection_results
    }

    // STEP 2: retrieve division times
    if (params.run_steps == "step2" || params.run_steps == "all") {
        if (!params.cell_type)                 { error "ERROR: --cell_type is required for step2" }
        if (!params.imaging_frequency)         { error "ERROR: --imaging_frequency is required for step2" }
        if (!params.time_interval_discrepancy) { error "ERROR: --time_interval_discrepancy is required for step2" }
        if (!params.colony_size_limit)         { error "ERROR: --colony_size_limit is required for step2" }

        if (!path_to_step1) {
            path_to_step1 = file("${params.outdir}/${exp_id}/${sub_id}/step1", type: 'dir')
        }
        if (!path_to_step0) {
            path_to_step0 = file("${params.outdir}/${exp_id}/${sub_id}/step0", type: 'dir')
        }

        STEP2_AGGREGATION(
            path_to_step1,
            exp_id,
            sub_id,
            params.cell_type,
            params.imaging_frequency,
            params.time_interval_discrepancy,
            params.colony_size_limit,
            path_to_step0,
            params.generate_qc_report
        )
    }
}

// ==========================================================
// PROCESS DEFINITIONS
// ==========================================================

process STEP0_PROCESS_CZI {
    tag "${exp_id}_${sub_id}"
    publishDir { "${params.outdir}/${exp_id}/${sub_id}" }, mode: 'copy'

    input:
    val czi_folder
    val exp_id
    val sub_id
    val rotation_angle

    output:
    path "step0", emit: folder

    script:
    """
    export PYTHONPATH="${launchDir}:${launchDir}/src"
    mkdir -p step0
    ${task.ext.python_bin} -m src.visiowell_pipeline.pipeline.step0_findwells \
        --folder_path ${czi_folder} --rotation_angle ${rotation_angle} --dir_results step0
    """
}

process STEP1_DETECTION {
    tag "${exp_id}_${sub_id}"
    publishDir { "${params.outdir}/${exp_id}/${sub_id}" }, mode: 'copy'

    input:
    path step0_output
    val exp_id
    val sub_id
    val which_network
    val name_experiment
    val training_to_load
    val flag_plot
    val flag_gt
    val positions
    val name_series_list
    val name_well_list

    output:
    path "step1", emit: detection_results

    script:
    def positions_arg = (positions && positions != "") ? "--positions ${positions}" : ""
    def series_arg    = (name_series_list && name_series_list != "") ? "--name_series_list ${name_series_list}" : ""
    def wells_arg     = (name_well_list && name_well_list != "") ? "--name_well_list ${name_well_list}" : ""
    def plot_arg = flag_plot ? "--flag_plot" : ""
    def gt_arg   = flag_gt   ? "--flag_gt" : ""

    """
    WD=\$PWD
    mkdir -p step1
    cd /app
    ${task.ext.python_bin} -m src.visiowell_pipeline.pipeline.step1_predictions \
        --path_exp \$WD/step0 \
        --dir_results \$WD/step1 \
        --which_network ${which_network} \
        --name_experiment ${name_experiment} \
        --training_to_load ${training_to_load} \
        ${plot_arg} \
        ${gt_arg} \
        ${positions_arg} \
        ${series_arg} \
        ${wells_arg}
    """
}
process STEP2_AGGREGATION {
    tag "${exp_id}_${sub_id}"
    publishDir { "${params.outdir}/${exp_id}/${sub_id}/final" }, mode: 'copy', pattern: "*.csv"
    publishDir { "${params.outdir}/${exp_id}/${sub_id}/qc_report" }, mode: 'copy', pattern: "*.txt", optional: true

    input:
    path step1_output
    val exp_id
    val sub_id
    val cell_type
    val imaging_frequency
    val time_interval_discrepancy
    val colony_size_limit
    path dir_wells
    val generate_qc_report

    output:
    path "*.csv", emit: final_csv
    path "*.txt", emit: qc_report, optional: true

    script:
    def qc_args = generate_qc_report ? "--dir_wells ${dir_wells} --qcreport_path ." : ""

    """
    export PYTHONPATH="${launchDir}:${launchDir}/src"

    ${task.ext.python_bin} -m src.visiowell_pipeline.pipeline.step2_output \
        --dir_results ${step1_output} \
        --dir_output . \
        --cell_type ${cell_type} \
        --imaging_frequency ${imaging_frequency} \
        --time_interval_discrepancy ${time_interval_discrepancy} \
        --colony_size_limit ${colony_size_limit} \
        --name_folder ${exp_id} \
        --name_scene ${sub_id} \
        ${qc_args}
    """
}
