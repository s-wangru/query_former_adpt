-- This is based on pgtune for pg10, commenting out max_parallel_workers.

-- WARNING
-- this tool not being optimal
-- for very high memory systems

-- DB Version: 10
-- OS Type: linux
-- DB Type: dw
-- Total Memory (RAM): 128 GB
-- CPUs num: 40
-- Data Storage: ssd

ALTER SYSTEM SET
 max_connections = '40';
ALTER SYSTEM SET
 shared_buffers = '32GB';
ALTER SYSTEM SET
 effective_cache_size = '96GB';
ALTER SYSTEM SET
 maintenance_work_mem = '2GB';
ALTER SYSTEM SET
 checkpoint_completion_target = '0.9';
ALTER SYSTEM SET
 wal_buffers = '16MB';
ALTER SYSTEM SET
 default_statistics_target = '500';
ALTER SYSTEM SET
 random_page_cost = '1.1';
ALTER SYSTEM SET
 effective_io_concurrency = '200';
ALTER SYSTEM SET
 work_mem = '20971kB';
ALTER SYSTEM SET
 huge_pages = 'try';
ALTER SYSTEM SET
 min_wal_size = '4GB';
ALTER SYSTEM SET
 max_wal_size = '16GB';
ALTER SYSTEM SET
 max_worker_processes = '40';
ALTER SYSTEM SET
 max_parallel_workers_per_gather = '20';
--ALTER SYSTEM SET
-- max_parallel_workers = '40';