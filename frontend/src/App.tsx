import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "./components/ActivityTimeline";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ChatMessagesView } from "./components/ChatMessagesView";

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);

  const thread = useStream<{
    messages: Message[];
    initial_search_query_count: number;
    max_research_loops: number;
    search_query_result_limit: number;
    reasoning_model: string;
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent",
    messagesKey: "messages",
    onFinish: (event: any) => {
      console.log(event);
    },
    onUpdateEvent: (event: any) => {
      console.log(event);
      let processedEvent: ProcessedEvent | null = null;
      if (event.generate_query && event.generate_query.research_query) {
        processedEvent = {
          title: "Generating Search Queries",
          data: event.generate_query.research_query,
        };
      } else if (event.reconcile_search) {
        const queries = event.recollection.search_query || [];
        const follow_ups = event.recollection.follow_up_queries || [];
        const numAdditionalSources = queries.length + follow_ups.length;
        processedEvent = {
          title: `Reconciling ${numAdditionalSources} additional queries`,
          data: event.reconcile_search.search_query.join(", "),
        };
      } else if (event.recollection) {
        const sources = event.recollection.sources_gathered || [];
        const numSources = sources.length;
        processedEvent = {
          title: `Recollected ${numSources} from past memory`,
          data: event.recollection.search_query,
        };
      } else if (event.web_research) {
        const sources = event.web_research.sources_gathered || [];
        const numSources = sources.length;
        const queryLabel = event.web_research.search_query;
        processedEvent = {
          title: "Web Research",
          data: `Gathered ${numSources} sources. Related to: ${
            queryLabel || "N/A"
          }.`,
        };
      } else if (event.reflection) {
        let follow_up_queries = null;
        if (event.reflection.follow_up_queries) {
          follow_up_queries = event.reflection.follow_up_queries.join(", ");
        } else {
          follow_up_queries = "follow up queries...";
        }

        processedEvent = {
          title: "Reflection",
          data: event.reflection.is_knowledge_sufficient
            ? "Search successful, generating final answer."
            : `Need more information, searching for ${follow_up_queries}`,
        };
      } else if (event.finalize_answer) {
        processedEvent = {
          title: "Finalizing Answer",
          data: "Composing and presenting the final answer.",
        };
        hasFinalizeEventOccurredRef.current = true;
      } else {
        processedEvent = null;
      }
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
  });

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      // convert effort to, initial_search_query_count and max_research_loops
      // low means max 1 loop and 1 query
      // medium means max 3 loops and 3 queries
      // high means max 10 loops and 5 queries
      let initial_search_query_count = 0;
      let max_research_loops = 0;
      let search_query_result_limit = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          search_query_result_limit = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          search_query_result_limit = 5;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          search_query_result_limit = 10;
          break;
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        search_query_result_limit: search_query_result_limit,
        reasoning_model: model,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="flex-1 flex flex-col overflow-hidden max-w-4xl mx-auto w-full">
        <div
          className={`flex-1 overflow-y-auto ${
            thread.messages.length === 0 ? "flex" : ""
          }`}
        >
          {thread.messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <ChatMessagesView
              messages={thread.messages}
              isLoading={thread.isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
            />
          )}
        </div>
      </main>
    </div>
  );
}
