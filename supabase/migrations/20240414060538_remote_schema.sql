create sequence "public"."documents_id_seq";

alter table "public"."documents" drop column "metadata";

alter table "public"."documents" alter column "id" set default nextval('documents_id_seq'::regclass);

alter table "public"."documents" alter column "id" set data type bigint using "id"::bigint;

alter sequence "public"."documents_id_seq" owned by "public"."documents"."id";

CREATE INDEX documents_embedding_idx ON public.documents USING ivfflat (embedding vector_cosine_ops) WITH (lists='100');

set check_function_bodies = off;

CREATE OR REPLACE FUNCTION public.match_documents(query_embedding vector, match_threshold double precision, match_count integer)
 RETURNS TABLE(id bigint, content text, similarity double precision)
 LANGUAGE sql
 STABLE
AS $function$
  select
    documents.id,
    documents.content,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.embedding <=> query_embedding < 1 - match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
$function$
;


create schema if not exists "vecs";

create table "vecs"."docs" (
    "id" character varying not null,
    "vec" vector(3) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


CREATE UNIQUE INDEX docs_pkey ON vecs.docs USING btree (id);

alter table "vecs"."docs" add constraint "docs_pkey" PRIMARY KEY using index "docs_pkey";


