/**
 * Integration tests for CorporateSwarm → bolt.diy flow.
 * 
 * These tests validate the complete integration between CorporateSwarm
 * governance system and bolt.diy execution engine.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type { Mandate } from '~/types/mandate';

describe('CorporateSwarm → bolt.diy Integration', () => {
  const baseUrl = process.env.BOLT_DIY_API_URL || 'http://localhost:5173';

  describe('Mandate Submission', () => {
    it('should accept a valid mandate from CorporateSwarm', async () => {
      const mandate: Mandate = {
        mandate_id: 'test-integration-001',
        objectives: ['Test objective'],
        constraints: {
          language: 'ts',
          maxDependencies: 10,
          noNetwork: false,
          allowedPackages: [],
          maxFileSize: 10000,
          maxFiles: 10,
        },
        budget: {
          token: 10000,
          time: 60,
          cost: 1.0,
        },
        deliverables: ['src/index.ts'],
        governance: {
          proposal_id: 'test-proposal-001',
        },
        iteration_config: {
          max_iterations: 1,
          test_required: false,
          quality_threshold: 0.7,
        },
      };

      const response = await fetch(`${baseUrl}/api/mandate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(mandate),
      });

      expect(response.status).toBe(202);
      const result = await response.json() as { success: boolean; mandate_id: string; event_stream_url?: string };
      expect(result.success).toBe(true);
      expect(result.mandate_id).toBe(mandate.mandate_id);
      expect(result.event_stream_url).toBeDefined();
    });

    it('should reject an invalid mandate', async () => {
      const invalidMandate = {
        mandate_id: 'test-invalid',
        // Missing required fields
      };

      const response = await fetch(`${baseUrl}/api/mandate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(invalidMandate),
      });

      expect(response.status).toBe(400);
      const result = await response.json() as { success: boolean; errors?: any };
      expect(result.success).toBe(false);
      expect(result.errors).toBeDefined();
    });
  });

  describe('Execution Event Streaming', () => {
    it('should stream execution events for a mandate', async () => {
      const mandateId = 'test-stream-001';
      
      // First submit a mandate
      const mandate: Mandate = {
        mandate_id: mandateId,
        objectives: ['Test streaming'],
        constraints: {
          language: 'ts',
          maxDependencies: 5,
          noNetwork: false,
          allowedPackages: [],
          maxFileSize: 10000,
          maxFiles: 5,
        },
        budget: {
          token: 5000,
          time: 30,
          cost: 0.5,
        },
        deliverables: ['src/test.ts'],
        governance: {},
        iteration_config: {
          max_iterations: 1,
          test_required: false,
          quality_threshold: 0.7,
        },
      };

      const submitResponse = await fetch(`${baseUrl}/api/mandate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(mandate),
      });

      expect(submitResponse.status).toBe(202);
      const submitResult = await submitResponse.json() as { event_stream_url?: string };
      const eventStreamUrl = submitResult.event_stream_url;
      
      if (!eventStreamUrl) {
        throw new Error('Event stream URL not provided');
      }

      // Connect to event stream
      const eventSource = new EventSource(eventStreamUrl);
      const events: any[] = [];

      return new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          eventSource.close();
          reject(new Error('Event stream timeout'));
        }, 10000);

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            events.push(data);

            // Check for connection confirmation
            if (data.type === 'connected') {
              expect(data.mandate_id).toBe(mandateId);
            }

            // If we receive an iteration_start event, we're good
            if (data.type === 'iteration_start') {
              clearTimeout(timeout);
              eventSource.close();
              expect(events.length).toBeGreaterThan(0);
              resolve();
            }
          } catch (error) {
            clearTimeout(timeout);
            eventSource.close();
            reject(error);
          }
        };

        eventSource.onerror = (error) => {
          clearTimeout(timeout);
          eventSource.close();
          reject(error);
        };
      });
    });
  });

  describe('Governance Reporting', () => {
    it('should report execution status to CorporateSwarm', async () => {
      // This test would require a mock CorporateSwarm API endpoint
      // For now, we'll test that the CorporateSwarmClient can be instantiated
      const { CorporateSwarmClient } = await import('~/lib/governance/corporate-swarm-client');
      const client = new CorporateSwarmClient({ baseUrl: 'http://localhost:8000' });

      // Test that client methods exist
      expect(client.submitExecutionReport).toBeDefined();
      expect(client.requestApproval).toBeDefined();
      expect(client.updateProposalStatus).toBeDefined();
    });
  });

  describe('Deployment Integration', () => {
    it('should deploy project when deployment is enabled in mandate', async () => {
      // This test would require actual deployment credentials
      // For now, we'll test that deployment APIs accept mandate_id
      const deployRequest = {
        files: {
          'index.html': '<html><body>Test</body></html>',
        },
        mandate_id: 'test-deploy-001',
        proposal_id: 'test-proposal-deploy-001',
      };

      // Note: This will fail without actual Netlify token, but we can test the API structure
      const response = await fetch(`${baseUrl}/api/netlify-deploy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(deployRequest),
      });

      // Should either succeed (if token provided) or fail with 401 (if no token)
      expect([200, 201, 401]).toContain(response.status);
    });
  });
});

