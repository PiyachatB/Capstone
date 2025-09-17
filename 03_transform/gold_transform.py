# gold_tables.py - Analytics-ready Gold layer tables
import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Fraud Detection and Risk Scoring
@dlt.table(
    name="capstone.schema_gold.claims_fraud_scores",
    comment="Claims with fraud risk scores and anomaly flags",
    table_properties={"delta.autoOptimize.optimizeWrite": "true"}
)
def gold_claims_fraud_scores():
    claims = dlt.read("capstone.schema_silver.claims")
    members = dlt.read("capstone.schema_silver.members")
    providers = dlt.read("capstone.schema_silver.providers")
    
    # Calculate fraud indicators
    return claims.join(members, "MemberID", "left") \
                 .join(providers, "ProviderID", "left") \
                 .withColumn("amount_zscore", 
                    (F.col("Amount") - F.avg("Amount").over(Window.partitionBy("ClaimType"))) /
                    F.stddev("Amount").over(Window.partitionBy("ClaimType"))
                 ) \
                 .withColumn("provider_claim_count_30d",
                    F.count("*").over(
                        Window.partitionBy("ProviderID")
                        .orderBy(F.unix_timestamp("ClaimDate").cast("bigint"))
                        .rangeBetween(-30*86400, 0)  # 30 days in seconds
                    )
                 ) \
                 .withColumn("member_claim_count_30d", 
                    F.count("*").over(
                        Window.partitionBy("MemberID")
                        .orderBy(F.unix_timestamp("ClaimDate").cast("bigint"))
                        .rangeBetween(-30*86400, 0)
                    )
                 ) \
                 .withColumn("fraud_score",
                    F.when(F.abs(F.col("amount_zscore")) > 3, 50)  # High amount anomaly
                     .when(F.col("provider_claim_count_30d") > 100, 40)  # High provider volume
                     .when(F.col("member_claim_count_30d") > 20, 30)  # High member volume
                     .when(F.col("Status") == "Rejected", 20)  # Rejection pattern
                     .otherwise(10)
                 ) \
                 .withColumn("risk_category",
                    F.when(F.col("fraud_score") >= 30, "High")
                     .when(F.col("fraud_score") >= 20, "Medium")
                     .otherwise("Low")
                 ) \
                 .select(
                    "ClaimID", "MemberID", "ProviderID", "ClaimDate", "Amount", "Status",
                    "fraud_score", "risk_category", "amount_zscore",
                    "provider_claim_count_30d", "member_claim_count_30d",
                    F.current_timestamp().alias("scored_timestamp")
                 )

# Claims Analytics Mart
@dlt.table(
    name="capstone.schema_gold.claims_analytics",
    comment="Pre-aggregated claims data for reporting dashboards"
)
def gold_claims_analytics():
    claims = dlt.read("capstone.schema_silver.claims")
    members = dlt.read("capstone.schema_silver.members")
    providers = dlt.read("capstone.schema_silver.providers")
    
    return claims.join(members, "MemberID", "left") \
                 .join(providers, "ProviderID", "left") \
                 .withColumn("claim_year", F.year("ClaimDate")) \
                 .withColumn("claim_month", F.month("ClaimDate")) \
                 .withColumn("days_to_service", 
                    F.datediff("ClaimDate", "ServiceDate")
                 ) \
                 .withColumn("icd10_primary", 
                    F.split("ICD10Codes", ";")[0]
                 ) \
                 .withColumn("cpt_primary",
                    F.split("CPTCodes", ";")[0]
                 ) \
                 .select(
                    "ClaimID", "MemberID", "ProviderID", F.col("providers.Name").alias("ProviderName"),
                    F.col("members.Name").alias("MemberName"), "Region", "PlanType",
                    "claim_year", "claim_month", "ClaimDate", "ServiceDate", 
                    "days_to_service", "Amount", "Status", "ClaimType",
                    "SubmissionChannel", "icd10_primary", "cpt_primary",
                    F.size(F.split("ICD10Codes", ";")).alias("diagnosis_count"),
                    F.size(F.split("CPTCodes", ";")).alias("procedure_count")
                 )

# Provider Performance Metrics
@dlt.table(
    name="capstone.schema_gold.provider_metrics",
    comment="Provider performance and quality metrics"
)
def gold_provider_metrics():
    claims = dlt.read("capstone.schema_silver.claims")
    providers = dlt.read("capstone.schema_silver.providers")
    
    provider_stats = claims.groupBy("ProviderID") \
        .agg(
            F.count("*").alias("total_claims"),
            F.sum("Amount").alias("total_amount"),
            F.avg("Amount").alias("avg_claim_amount"),
            F.sum(F.when(F.col("Status") == "Approved", 1).otherwise(0)).alias("approved_claims"),
            F.sum(F.when(F.col("Status") == "Rejected", 1).otherwise(0)).alias("rejected_claims"),
            F.sum(F.when(F.col("Status") == "Pending", 1).otherwise(0)).alias("pending_claims"),
            F.countDistinct("MemberID").alias("unique_members"),
            F.max("ClaimDate").alias("last_claim_date")
        ) \
        .withColumn("approval_rate", 
            F.col("approved_claims") / F.col("total_claims")
        ) \
        .withColumn("rejection_rate",
            F.col("rejected_claims") / F.col("total_claims") 
        )
    
    return providers.join(provider_stats, "ProviderID", "left") \
                   .withColumn("performance_score",
                      F.when(F.col("approval_rate") > 0.9, "Excellent")
                       .when(F.col("approval_rate") > 0.8, "Good") 
                       .when(F.col("approval_rate") > 0.7, "Fair")
                       .otherwise("Poor")
                   ) \
                   .select(
                      "ProviderID", "Name", "Specialties", "State",
                      "total_claims", "total_amount", "avg_claim_amount",
                      "approved_claims", "rejected_claims", "pending_claims",
                      "approval_rate", "rejection_rate", "unique_members",
                      "performance_score", "last_claim_date"
                   )

# Member Utilization Summary 
@dlt.table(
    name="capstone.schema_gold.member_utilization",
    comment="Member healthcare utilization patterns and costs"
)
def gold_member_utilization():
    claims = dlt.read("capstone.schema_silver.claims")
    members = dlt.read("capstone.schema_silver.members")

    
    member_stats = claims.groupBy("MemberID") \
        .agg(
            F.count("*").alias("total_claims"),
            F.sum("Amount").alias("total_cost"),
            F.avg("Amount").alias("avg_claim_cost"),
            F.countDistinct("ProviderID").alias("unique_providers"),
            F.countDistinct("ClaimType").alias("service_types"),
            F.min("ClaimDate").alias("first_claim_date"),
            F.max("ClaimDate").alias("last_claim_date"),
            F.sum(F.when(F.col("ClaimType") == "Inpatient", F.col("Amount")).otherwise(0)).alias("inpatient_cost"),
            F.sum(F.when(F.col("ClaimType") == "Outpatient", F.col("Amount")).otherwise(0)).alias("outpatient_cost"),
            F.sum(F.when(F.col("ClaimType") == "Pharmacy", F.col("Amount")).otherwise(0)).alias("pharmacy_cost")
        ) \
        .withColumn("utilization_tier",
            F.when(F.col("total_cost") > 10000, "High")
             .when(F.col("total_cost") > 5000, "Medium")
             .otherwise("Low")
        )
    
    return members.join(member_stats, "MemberID", "left") \
                  .select(
                     "MemberID", "Name", "DOB", "Gender", "Region", "PlanType",
                     "total_claims", "total_cost", "avg_claim_cost",
                     "unique_providers", "service_types", "utilization_tier",
                     "inpatient_cost", "outpatient_cost", "pharmacy_cost",
                     "first_claim_date", "last_claim_date"
                  )

# Compliance and Audit Trail
@dlt.table(
    name="capstone.schema_gold.compliance_audit",
    comment="Audit trail and compliance metrics for regulatory reporting"
)
def gold_compliance_audit():
    claims = dlt.read("capstone.schema_silver.claims")
    members = dlt.read("capstone.schema_silver.members")
    
    return claims.join(members, "MemberID", "left") \
                 .withColumn("processing_delay_days",
                    F.datediff("ProcessedTimestamp", "ClaimDate")
                 ) \
                 .withColumn("compliance_flag",
                    F.when(F.col("processing_delay_days") > 30, "DELAYED_PROCESSING")
                     .when(F.col("Amount") > 50000, "HIGH_VALUE_CLAIM")
                     .when(F.col("Email").isNull(), "MISSING_CONTACT")
                     .otherwise("COMPLIANT")
                 ) \
                 .withColumn("audit_category",
                    F.when(F.col("compliance_flag") != "COMPLIANT", "REVIEW_REQUIRED")
                     .otherwise("STANDARD")
                 ) \
                 .select(
                    "ClaimID", "MemberID", "ClaimDate", "Amount", "Status",
                    "SourceType", "ProcessedTimestamp", "processing_delay_days",
                    "compliance_flag", "audit_category",
                    F.current_timestamp().alias("audit_timestamp")
                 )

# Monthly Claims Summary (Aggregated)
@dlt.table(
    name="capstone.schema_gold.monthly_summary",
    comment="Monthly aggregated claims metrics for executive reporting"
)
def gold_monthly_summary():
    claims = dlt.read("capstone.schema_silver.claims")
    
    return claims.withColumn("claim_year", F.year("ClaimDate")) \
                 .withColumn("claim_month", F.month("ClaimDate")) \
                 .groupBy("claim_year", "claim_month", "Status", "ClaimType") \
                 .agg(
                    F.count("*").alias("claim_count"),
                    F.sum("Amount").alias("total_amount"),
                    F.avg("Amount").alias("avg_amount"),
                    F.min("Amount").alias("min_amount"),
                    F.max("Amount").alias("max_amount"),
                    F.countDistinct("MemberID").alias("unique_members"),
                    F.countDistinct("ProviderID").alias("unique_providers")
                 ) \
                 .withColumn("amount_per_member",
                    F.col("total_amount") / F.col("unique_members")
                 ) \
                 .withColumn("summary_date", F.current_date()) \
                 .orderBy("claim_year", "claim_month", "Status", "ClaimType")