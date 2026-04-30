import { type NodeFetchers, type FetcherType } from '../../fetchers';
import { ActivationDisplay } from '../ActivationDisplay';

interface ReluChannelNodeProps {
  channelIndex: number;
  layerId: string;
  fetchers?: NodeFetchers;
  fetcherType?: FetcherType;
}

export const ReluChannelNodeData = ({ channelIndex, layerId, fetchers, fetcherType = "activation" }: ReluChannelNodeProps) => {
  const coordinate = `${layerId}.out_${channelIndex}`;
  const fetcher = fetchers?.[fetcherType];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>

      {/* ── Toolbar ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 5,
        padding: '4px 8px',
        background: 'rgba(0, 0, 0, 0.55)',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '5px 5px 0 0',
      }}>
        <span style={{
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          color: '#fbbf24',
          lineHeight: 1,
        }}>
          ReLU
        </span>
        <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: 9, lineHeight: 1 }}>·</span>
        <span style={{
          fontSize: 9,
          fontWeight: 500,
          color: 'rgba(255,255,255,0.65)',
          lineHeight: 1,
        }}>
          Ch {channelIndex}
        </span>
      </div>

      {/* ── Activation map (main area) ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 6,
        background: '#0d0d14',
        borderRadius: '0 0 5px 5px',
      }}>
        {fetcher ? (
          <ActivationDisplay
            coordinate={coordinate}
            fetcher={fetcher}
            maxSize={84}
          />
        ) : (
          <div style={{
            width: 84,
            height: 84,
            background: '#1e1e2e',
            borderRadius: 4,
            border: '1px solid rgba(255,255,255,0.08)',
          }} />
        )}
      </div>

    </div>
  );
};