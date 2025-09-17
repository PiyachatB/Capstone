import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
# Silver layer tables with data quality rules




@dlt.table(
    name="capstone.schema_silver.claims",
    comment="Cleaned and merged claims data from batch and streaming sources",
    table_properties={"delta.autoOptimize.optimizeWrite": "true"}
)
@dlt.expect_all_or_drop({
    "valid_claim_id": "ClaimID IS NOT NULL AND ClaimID != ''",
    "valid_member_id": "MemberID IS NOT NULL AND MemberID != ''", 
    "valid_provider_id": "ProviderID IS NOT NULL AND ProviderID != ''",
    "valid_amount": "Amount > 0",
    "valid_status": "Status IN ('Submitted', 'Pending', 'Approved', 'Rejected')"
})
def silver_claims():
    # Read bronze batch claims
    batch_claims = dlt.read("capstone.schema_bronze.claims_batch").select(
        "ClaimID", "MemberID", "ProviderID", "ClaimDate", "ServiceDate", 
        "Amount", "Status", "ICD10Codes", "CPTCodes", "ClaimType", 
        "SubmissionChannel", "Notes",
        F.col("IngestTimestamp").alias("ProcessedTimestamp"),
        F.col("_source").alias("SourceType")
    )
    
    # Read bronze stream claims
    stream_claims = dlt.read("capstone.schema_bronze.claims_stream").select(
        "ClaimID", "MemberID", "ProviderID", "ClaimDate", 
        F.lit(None).cast(StringType()).alias("ServiceDate"),
        "Amount", "Status", "ICD10Codes", "CPTCodes",
        F.lit(None).cast(StringType()).alias("ClaimType"),
        F.lit(None).cast(StringType()).alias("SubmissionChannel"),
        F.lit(None).cast(StringType()).alias("Notes"),
        F.col("EventTimestamp").alias("ProcessedTimestamp"),
        F.col("_source").alias("SourceType")
    )
    
    # Union and deduplicate (keep latest by ProcessedTimestamp)
    return batch_claims.union(stream_claims) \
        .withColumn("ClaimDate", F.to_date("ClaimDate")) \
        .withColumn("ServiceDate", F.to_date("ServiceDate")) \
        .withColumn("ProcessedTimestamp", F.to_timestamp("ProcessedTimestamp")) \
        .withColumn("row_num", 
            F.row_number().over(
                Window.partitionBy("ClaimID")
                .orderBy(F.desc("ProcessedTimestamp"))
            )
        ) \
        .filter("row_num = 1") \
        .drop("row_num")

@dlt.table(
    name="capstone.schema_silver.members",
    comment="Cleaned member master data with data quality checks"
)
@dlt.expect_all_or_drop({
    "valid_member_id": "MemberID IS NOT NULL AND MemberID != ''",
    "valid_name": "Name IS NOT NULL AND Name != ''",
    "valid_dob": "DOB IS NOT NULL",
    "valid_gender": "Gender IN ('M', 'F', 'O')",
    "valid_plan": "PlanType IN ('HMO', 'EPO', 'Medicare', 'Medicaid')"
})
def silver_members():
    return dlt.read("capstone.schema_bronze.members") \
        .withColumn("DOB", F.to_date("DOB")) \
        .withColumn("EffectiveDate", F.to_date("EffectiveDate")) \
        .withColumn("LastUpdated", F.to_date("LastUpdated")) \
        .withColumn("IsActive", F.col("IsActive").cast(BooleanType())) \
        .withColumn("Email", F.lower(F.trim("Email")))

@dlt.table(
    name="capstone.schema_silver.providers", 
    comment="Cleaned provider directory with flattened locations"
)
@dlt.expect_all_or_drop({
    "valid_provider_id": "ProviderID IS NOT NULL AND ProviderID != ''",
    "valid_name": "Name IS NOT NULL AND Name != ''",
    "valid_tin": "TIN IS NOT NULL AND length(TIN) = 9"
})
def silver_providers():
    return dlt.read("capstone.schema_bronze.providers") \
        .withColumn("LastVerified", F.to_date("LastVerified")) \
        .withColumn("Specialties", F.col("Specialties").cast(ArrayType(StringType()))) \
        .withColumn("Location", F.explode("Locations")) \
        .select(
            "ProviderID", "Name", "Specialties", "IsActive", "TIN", "LastVerified",
            F.col("Location.Address").alias("Address"),
            F.col("Location.City").alias("City"), 
            F.col("Location.State").alias("State")
        ) \
        .withColumn("row_num", 
            F.row_number().over(
                Window.partitionBy("ProviderID")
                .orderBy("Address")
            )
        ) \
        .filter("row_num = 1") \
        .drop("row_num")

@dlt.table(
    name="capstone.schema_silver.diagnosis_ref",
    comment="Cleaned diagnosis reference data"
)
@dlt.expect_all_or_drop({
    "valid_code": "Code IS NOT NULL AND Code != ''",
    "valid_description": "Description IS NOT NULL AND Description != ''"
})
def silver_diagnosis_ref():
    return dlt.read("capstone.schema_bronze.diagnosis_ref") \
        .withColumn("Code", F.upper(F.trim("Code"))) \
        .withColumn("Description", F.trim("Description"))

# Data quality monitoring
@dlt.table(
    name="capstone.schema_log.data_quality_metrics"
)
def silver_data_quality_metrics():
    claims_count = dlt.read("capstone.schema_silver.claims").count()
    members_count = dlt.read("capstone.schema_silver.members").count()
    providers_count = dlt.read("capstone.schema_silver.providers").count()
    
    return spark.createDataFrame([
        ("claims", claims_count),
        ("members", members_count), 
        ("providers", providers_count)
    ], ["table_name", "record_count"]) \
    .withColumn("processed_timestamp", F.current_timestamp())

